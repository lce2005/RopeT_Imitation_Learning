"""Milestone 3 (part 1): demonstrations -> (observation, action) training pairs.

Given (do not change):  load_demos, split_by_episode
TODO (you):             make_features, make_pairs, Normalizer, BCDataset

The policy sees positions only: rope keypoints, goal keypoints, stick tip (all in each episode's 'obs', see
sim/scene.py OBS_LAYOUT), and predicts the stick-tip velocity 'act' (3,) in [-1, 1].
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sim'))
from episode_io import load, problems          # noqa: E402
from scene import OBS_LAYOUT                    # noqa: E402


# ---------------------------------------------------------------- given
PER_STEP = ('obs', 'act', 'human', 'kp', 'kp_z', 'tip', 'err', 'succ', 'qpos')     # the keys with one row per step


def load_demos(folders, drop_idle=True, only_success=False):
    """Every valid ep_*.npz in the folders, as episode dicts (sim/episode_io.py keys). drop_idle: remove the steps where
    no key was held (the operator thinking is not an action to imitate). only_success: keep successful episodes only."""
    eps = []
    for d in folders:
        for f in sorted(Path(d).glob('ep_*.npz')):
            ep = load(f)
            if problems(ep) or (only_success and not ep['success']): continue
            if drop_idle:
                h = ep['human'].astype(bool)
                ep = {k: (v[h] if k in PER_STEP else v) for k, v in ep.items()}
                ep['length'] = int(h.sum())
            if ep['length'] > 0: ep['name'] = f.stem; eps.append(ep)
    return eps


def split_by_episode(eps, val_frac=.2, seed=0):
    """(train, val) lists of episodes. Split by episode, not by step: neighbouring steps are almost the same, so a
    step-wise split puts near-copies of the training data into validation."""
    i = np.random.default_rng(seed).permutation(len(eps)); n_val = max(1, int(round(len(eps) * val_frac))) if len(eps) > 1 else 0
    return [eps[j] for j in i[n_val:]], [eps[j] for j in i[:n_val]]

# ---------------------------------------------------------------- TODO 구현
# 기존 파일 위쪽의 import numpy as np, OBS_LAYOUT import는 그대로 사용
from torch.utils.data import Dataset


def make_features(obs):
    """
    관측 기록 (T, 85)을 모델에 줄 위치 특징 (T, 77)로 바꾼다.

    T는 기록된 스텝 수다.
    특징 77개는 다음 순서로 배치한다.
      1. 로프 각 점 - 막대 끝의 x, y 좌표: 38개
      2. 목표 각 점 - 같은 번호의 로프 점: 38개
      3. 막대 끝의 높이 z: 1개
    """
    obs = np.asarray(obs, dtype=np.float32)
    num_steps = obs.shape[0]

    # OBS_LAYOUT은 관측 85개 중 각 정보가 어느 열에 있는지 알려준다.
    # reshape하면 각 스텝마다 로프 점 19개의 (x, y)를 구분할 수 있다.
    rope_xy = obs[:, OBS_LAYOUT["rope_kp_xy"]].reshape(num_steps, 19, 2)
    goal_xy = obs[:, OBS_LAYOUT["goal_kp_xy"]].reshape(num_steps, 19, 2)
    tip_xyz = obs[:, OBS_LAYOUT["tip_xyz"]]  # (T, 3)

    # 막대 끝의 x, y를 (T, 1, 2) 모양으로 만든다.
    # NumPy가 스텝마다 같은 막대 좌표를 로프 점 19개에서 각각 빼준다.
    # 결과의 모양은 (T, 19, 2)다.
    tip_to_rope = rope_xy - tip_xyz[:, None, :2]

    # 같은 번호의 로프 점이 목표까지 얼마나 이동해야 하는지 계산한다.
    # 예를 들어 x 차이가 양수라면 목표점이 로프 점보다 +x 쪽에 있다.
    rope_to_goal = goal_xy - rope_xy

    # 2:3으로 선택하면 (T, 1) 모양이 유지되어 아래에서 붙이기 쉽다.
    tip_height = tip_xyz[:, 2:3]

    # 점 19개의 (dx, dy)를 스텝마다 숫자 38개로 펼친다.
    # axis=1은 각 스텝의 정보들을 열 방향으로 이어 붙인다는 뜻이다.
    features = np.concatenate(
        [
            tip_to_rope.reshape(num_steps, 38),
            rope_to_goal.reshape(num_steps, 38),
            tip_height,
        ],
        axis=1,
    )

    return features.astype(np.float32)


def make_pairs(eps, obs_horizon=1, act_horizon=1):
    """
    여러 에피소드에서 모델 입력 X와 사람 행동 정답 Y를 만든다.

    obs_horizon: 한 입력에 포함할 관측 스텝 수
    act_horizon: 한 번에 예측할 행동 스텝 수

    v0: 관측 1개, 행동 1개 → X (N, 77),  Y (N, 3)
    v1: 관측 4개, 행동 1개 → X (N, 308), Y (N, 3)
    v2: 관측 1개, 행동 4개 → X (N, 77),  Y (N, 12)

    N은 모든 에피소드의 스텝 수를 합친 값이다.
    """
    if obs_horizon < 1 or act_horizon < 1:
        raise ValueError("관측 창과 행동 창은 각각 1 이상이어야 합니다.")

    all_inputs = []
    all_targets = []

    # 한 에피소드 안에서 창을 만든 뒤 마지막에 합친다.
    # 이렇게 해야 이전 판의 마지막 관측이 다음 판의 입력에 섞이지 않는다.
    for episode in eps:
        features = make_features(episode["obs"])
        actions = np.asarray(episode["act"], dtype=np.float32)
        num_steps = len(features)

        if actions.shape != (num_steps, 3):
            raise ValueError("행동 데이터는 관측과 길이가 같고 (T, 3) 모양이어야 합니다.")
        if num_steps == 0:
            continue

        # 각 행이 학습 샘플 하나의 기준 시점 t를 뜻한다.
        # 예를 들어 스텝이 3개라면 [[0], [1], [2]]가 된다.
        times = np.arange(num_steps)[:, None]

        # 관측 창이 4라면 [-3, -2, -1, 0]을 기준 시점에 더한다.
        # 따라서 시점 t의 입력은 t-3, t-2, t-1, t의 관측이다.
        obs_offsets = np.arange(-(obs_horizon - 1), 1)
        obs_indices = times + obs_offsets

        # 에피소드 시작에 과거 관측이 부족하면 첫 관측을 반복한다.
        # 예: t=1, 관측 창=4 → [-2, -1, 0, 1] → [0, 0, 0, 1]
        obs_indices = np.clip(obs_indices, 0, num_steps - 1)

        # 행동 창이 4라면 현재부터 t, t+1, t+2, t+3의 행동을 정답으로 쓴다.
        act_offsets = np.arange(act_horizon)
        act_indices = times + act_offsets

        # 에피소드 끝에서 미래 행동이 부족하면 마지막 행동을 반복한다.
        # 교수님 make_pairs() 명세에 맞춘 처리다.
        act_indices = np.clip(act_indices, 0, num_steps - 1)

        # features[obs_indices]의 모양은 (T, 관측 창, 77)이다.
        # 각 행에서 오래된 관측부터 현재 관측까지 차례대로 펼친다.
        inputs = features[obs_indices].reshape(
            num_steps, obs_horizon * features.shape[1]
        )

        # actions[act_indices]의 모양은 (T, 행동 창, 3)이다.
        # 각 행을 [현재 vx, vy, vz, 다음 vx, vy, vz, ...]로 펼친다.
        targets = actions[act_indices].reshape(num_steps, act_horizon * 3)

        all_inputs.append(inputs)
        all_targets.append(targets)

    if not all_inputs:
        raise ValueError("입력과 정답을 만들 수 있는 시연이 없습니다.")

    # axis=0으로 합치면 에피소드별 샘플이 행 방향으로 이어진다.
    X = np.concatenate(all_inputs, axis=0).astype(np.float32)
    Y = np.concatenate(all_targets, axis=0).astype(np.float32)
    return X, Y


class Normalizer:
    """
    각 열을 평균 0, 표준편차 약 1이 되도록 표준화한다.

    입력 X와 행동 Y에는 각각 별도의 Normalizer를 사용한다.
    fit()은 train 데이터에만 호출하고,
    검증 및 실제 조종에서는 저장된 평균과 표준편차를 사용한다.
    """

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("정규화 학습에는 비어 있지 않은 2차원 배열이 필요합니다.")

        # axis=0은 행들을 따라 계산해 각 열의 평균/표준편차를 구한다.
        # float64로 계산해 누적 계산 오차를 줄이고 float32로 저장한다.
        self.mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = x.std(axis=0, dtype=np.float64).astype(np.float32)

        # 값이 항상 같은 열은 표준편차가 0이다.
        # 0으로 나누는 것을 막기 위해 아주 작은 표준편차는 1로 바꾼다.
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return self

    def __call__(self, x):
        # normalizer(x)라고 호출하면 이 함수가 실행된다.
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, y):
        # 표준화된 모델 출력을 원래 행동 스케일로 되돌린다.
        y = np.asarray(y, dtype=np.float32)
        return (y * self.std + self.mean).astype(np.float32)

    def state_dict(self):
        # 체크포인트에 평균과 표준편차를 저장하기 위한 딕셔너리다.
        # list로 변환하면 JSON으로도 저장할 수 있다.
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_state(cls, d):
        # 저장된 정보로 정규화 객체를 복원한다.
        # 평가 데이터에서 평균과 표준편차를 다시 계산할 필요가 없다.
        normalizer = cls()
        normalizer.mean = np.asarray(d["mean"], dtype=np.float32)
        normalizer.std = np.asarray(d["std"], dtype=np.float32)
        return normalizer


class BCDataset(Dataset):
    """
    PyTorch가 입력과 정답을 한 쌍씩 꺼낼 수 있도록 만든 데이터셋이다.

    ds[i] → (i번째 표준화 입력, i번째 표준화 정답)
    DataLoader가 이 쌍들을 모아 배치를 만든다.
    """

    def __init__(self, X, Y, x_norm, y_norm):
        import torch

        super().__init__()

        if len(X) != len(Y):
            raise ValueError("입력 X와 정답 Y의 샘플 수가 같아야 합니다.")

        # 전달받은 정규화 객체를 이용해 표준화한다.
        # 여기서는 평균과 표준편차를 새로 계산하지 않는다.
        # torch.tensor는 신경망이 사용할 float32 텐서를 만든다.
        self.X = torch.tensor(x_norm(X), dtype=torch.float32)
        self.Y = torch.tensor(y_norm(Y), dtype=torch.float32)

    def __len__(self):
        # len(ds)가 전체 학습 샘플 수를 반환하게 한다.
        return len(self.X)

    def __getitem__(self, index):
        # ds[index]가 같은 시점의 입력과 정답을 반환하게 한다.
        return self.X[index], self.Y[index]
