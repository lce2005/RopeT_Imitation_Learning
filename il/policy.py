"""Milestone 3 (part 2): the policy network and the agent that runs it in the simulator.

Given (do not change):  save_checkpoint, load_agent, the Agent interface (reset, act)
TODO (you):             MLPPolicy, Agent.act

Checkpoint (torch.save dict): model (state_dict), config (the MLPPolicy kwargs + obs_horizon, act_horizon, ...),
x_norm / y_norm (Normalizer.state_dict()). eval_policy.py only calls load_agent(path) -> Agent.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from dataset import make_features, Normalizer


class MLPPolicy(nn.Module):
    """
    표준화된 위치 특징을 받아 표준화된 행동을 예측하는 신경망.

    B는 한 번에 처리하는 샘플 수인 배치 크기다.

    v0: 입력 (B, 77)  → 출력 (B, 3)
    v1: 입력 (B, 308) → 출력 (B, 3)
    v2: 입력 (B, 77)  → 출력 (B, 12)

    in_dim과 out_dim은 train_bc.py가 데이터 크기를 보고 전달한다.
    """

    def __init__(self, in_dim, out_dim, hidden=256, layers=3):
        super().__init__()

        if in_dim < 1 or out_dim < 1 or hidden < 1 or layers < 1:
            raise ValueError("입출력 크기, 은닉층 크기, 은닉층 수는 1 이상이어야 합니다.")

        network_layers = []
        current_dim = in_dim

        # layers=3, hidden=256이라면 은닉층을 256개 뉴런씩 3개 만든다.
        # 각 은닉층은 Linear와 GELU로 구성된다.
        #
        # Linear: 입력에 학습 가능한 가중치와 편향을 적용한다.
        # GELU: 비선형 변환을 적용해 복잡한 위치-행동 관계를 학습하게 한다.
        for _ in range(layers):
            network_layers.append(nn.Linear(current_dim, hidden))
            network_layers.append(nn.GELU())
            current_dim = hidden

        # 마지막 층은 필요한 행동 숫자만큼 출력한다.
        # 행동 창이 1이면 3개, 행동 창이 4이면 12개다.
        #
        # 출력은 표준화된 행동이므로 여기서는 [-1, 1]로 제한하지 않는다.
        # 실제 조종할 때 원래 스케일로 되돌린 후 범위를 제한한다.
        network_layers.append(nn.Linear(current_dim, out_dim))

        # Sequential은 등록된 층들을 위에서 정한 순서대로 실행한다.
        self.network = nn.Sequential(*network_layers)

    def forward(self, x):
        # model(x)를 호출하면 이 함수가 실행된다.
        # 입력 x의 모양은 (B, in_dim), 결과는 (B, out_dim)이다.
        return self.network(x)


class Agent:
    """
    시뮬레이터에서 학습된 모델을 이용해 실제 행동을 선택한다.

    reset(): 새 에피소드가 시작될 때 호출된다.
    act(obs): 현재 관측 (85,)를 받고 행동 (3,)을 반환한다.

    v2에서도 시뮬레이터에는 매번 행동 하나만 전달한다.
    모델이 예측한 4개 행동은 내부에 보관하고 순서대로 꺼낸다.
    """

    def __init__(self, model, config, x_norm, y_norm):
        # eval()은 모델을 평가 모드로 바꾼다.
        self.model = model.eval()
        self.cfg = config

        # 학습 데이터에서 계산해 체크포인트에 저장한 정규화 정보다.
        self.x_norm = x_norm
        self.y_norm = y_norm

        # To는 관측 창, Ta는 행동 창이다.
        self.To = int(config.get("obs_horizon", 1))
        self.Ta = int(config.get("act_horizon", 1))

        if self.To < 1 or self.Ta < 1:
            raise ValueError("관측 창과 행동 창은 각각 1 이상이어야 합니다.")

        # hist는 최근 To개 관측의 특징을 보관한다.
        # 가득 찬 상태에서 새 특징을 넣으면 가장 오래된 특징이 빠진다.
        self.hist = deque(maxlen=self.To)

        # plan에는 예측했지만 아직 실행하지 않은 행동들을 보관한다.
        self.plan = []

    def reset(self):
        # 이전 에피소드의 관측과 행동 계획이 다음 판에 섞이지 않게 비운다.
        self.hist.clear()
        self.plan = []

    def observe(self, obs):
        """Append one observation to the policy history."""
        obs = np.asarray(obs, dtype=np.float32)

        # 시뮬레이터는 관측 하나를 (85,) 모양으로 준다.
        # make_features()는 (T, 85)를 받으므로 (1, 85)로 바꿔 전달한다.
        # 반환값 (1, 77)의 첫 행을 꺼내면 현재 특징 (77,)이 된다.
        features = make_features(obs[None, :])[0]

        if len(self.hist) == 0:
            # 첫 스텝에는 과거 관측이 없다.
            # dataset.py의 make_pairs()처럼 첫 관측을 반복해 창을 채운다.
            # 예: To=4이면 [첫 관측, 첫 관측, 첫 관측, 첫 관측]
            for _ in range(self.To):
                self.hist.append(features.copy())
        else:
            # 오래된 관측부터 현재 관측 순서로 보관한다.
            self.hist.append(features.copy())

    def policy_input(self):
        """Build the normalized input from the current observation history."""
        window = np.concatenate(list(self.hist), axis=0)
        device = next(self.model.parameters()).device
        return torch.as_tensor(self.x_norm(window), dtype=torch.float32, device=device)

    @torch.no_grad()
    def act(self, obs):
        self.observe(obs)
        # 행동 묶음을 실행하는 동안에도 위에서 관측 기록은 계속 갱신한다.
        # 새 행동 예측은 기존 계획을 모두 실행했을 때 수행한다.
        if not self.plan:
            model_input = self.policy_input()[None, :]

            prediction = self.model(model_input)[0].cpu().numpy()

            # 학습된 행동 정규화를 되돌려 실제 행동 스케일로 복원한다.
            actions = self.y_norm.inverse(prediction)

            # Ta=4라면 12개 숫자를 행동 4개의 (vx, vy, vz)로 나눈다.
            actions = actions.reshape(self.Ta, 3)

            # 시뮬레이터가 받는 행동 범위에 맞춘다.
            actions = np.clip(actions, -1.0, 1.0).astype(np.float32)

            # 현재부터 실행할 순서대로 저장한다.
            self.plan = [action.copy() for action in actions]

        # 가장 앞의 행동 하나를 꺼내고 목록에서는 제거한다.
        # Ta=1이면 매 스텝 새로 예측한다.
        # Ta=4이면 4개를 차례로 실행한 뒤 다음 호출에서 새로 예측한다.
        return self.plan.pop(0)


# ---------------------------------------------------------------- given
def save_checkpoint(path, model, config, x_norm, y_norm, **extra):
    torch.save(dict(model=model.state_dict(), config=config, x_norm=x_norm.state_dict(), y_norm=y_norm.state_dict(), **extra), path)


def load_agent(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = MLPPolicy(**ck['config']['model']); model.load_state_dict(ck['model'])
    return Agent(model, ck['config'], Normalizer.from_state(ck['x_norm']), Normalizer.from_state(ck['y_norm']))
