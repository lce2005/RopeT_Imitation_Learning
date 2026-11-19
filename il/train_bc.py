#!/usr/bin/env python3
"""Milestone 3 (part 3): behaviour cloning on your demonstrations.

    python il/train_bc.py --data demos/20261234 --out runs/m3            # after the TODOs in dataset.py and policy.py
    python il/train_bc.py --data demos/20261234 --out runs/m3 --dry-run  # one batch forward + backward, no training

Given: argument parsing, data loading and the split, the checkpoint and the log format.
Training uses mean squared error on normalized actions.
Writes OUT/ckpt.pt (policy.load_agent reads it), OUT/train_log.csv (epoch, train_loss, val_loss), OUT/config.json.
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch

from dataset import load_demos, split_by_episode, make_pairs, Normalizer, BCDataset
from policy import MLPPolicy, save_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True, help='demo folders (ep_*.npz)')
    ap.add_argument('--out', type=Path, default=Path('runs/m3'))
    ap.add_argument('--epochs', type=int, default=200); ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--obs-horizon', type=int, default=1); ap.add_argument('--act-horizon', type=int, default=1)
    ap.add_argument('--only-success', action='store_true'); ap.add_argument('--keep-idle', action='store_true')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    # ---- data (given)
    eps = load_demos(a.data, drop_idle=not a.keep_idle, only_success=a.only_success)
    tr, va = split_by_episode(eps, .2, a.seed)
    Xtr, Ytr = make_pairs(tr, a.obs_horizon, a.act_horizon)
    Xva, Yva = make_pairs(va, a.obs_horizon, a.act_horizon)
    print(f'{len(eps)} episodes ({len(Xtr) + len(Xva)} base samples): {len(tr)} train, {len(va)} val')
    x_norm, y_norm = Normalizer(), Normalizer(); x_norm.fit(Xtr); y_norm.fit(Ytr)
    dl_tr = torch.utils.data.DataLoader(BCDataset(Xtr, Ytr, x_norm, y_norm), batch_size=a.batch, shuffle=True)
    dl_va = torch.utils.data.DataLoader(BCDataset(Xva, Yva, x_norm, y_norm), batch_size=a.batch)
    config = dict(model=dict(in_dim=Xtr.shape[1], out_dim=Ytr.shape[1], hidden=a.hidden, layers=a.layers),
                  obs_horizon=a.obs_horizon, act_horizon=a.act_horizon, data=a.data, n_train=len(Xtr), n_val=len(Xva),
                  train_episodes=[e['name'] for e in tr], val_episodes=[e['name'] for e in va], args={k: str(v) for k, v in vars(a).items()})
    (a.out / 'config.json').write_text(json.dumps(config, indent=2))
    model = MLPPolicy(**config['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    print(f'X {Xtr.shape}, Y {Ytr.shape}; {sum(p.numel() for p in model.parameters())} parameters')

    if a.dry_run:      # one batch, to check shapes and gradients before training
        x, y = next(iter(dl_tr)); loss = torch.nn.functional.mse_loss(model(x), y); loss.backward(); opt.step()
        print(f'dry run: batch {tuple(x.shape)} -> {tuple(model(x).shape)}, loss {loss.item():.4f}; OK'); return

    # ---- training
    # 손실은 예측한 행동과 실제 시연 행동의 차이를 나타낸다.
    # 여기서는 표준화된 행동끼리 MSE를 계산한다.
    loss_fn = torch.nn.MSELoss()

    # 검증 손실이 가장 낮았던 값을 기록한다.
    # 처음에는 무한대로 설정해 첫 에포크의 모델도 저장할 수 있게 한다.
    best = float("inf")
    t0 = time.time()

    # with를 사용하면 학습이 끝나거나 오류가 발생했을 때 파일이 닫힌다.
    with open(a.out / "train_log.csv", "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "train_loss", "val_loss", "time_s"])

        # 에포크 하나는 학습 데이터 전체를 한 번씩 사용하는 과정이다.
        for epoch in range(a.epochs):

            # 1. 학습
            model.train()

            train_loss_sum = 0.0
            train_count = 0

            # DataLoader가 입력과 정답을 배치 단위로 꺼내준다.
            # x는 위치 특징, y는 사람이 입력한 행동이다.
            for x, y in dl_tr:
                # 이전 배치에서 계산한 기울기를 지운다.
                # PyTorch는 기본적으로 기울기를 누적하므로 매번 초기화한다.
                opt.zero_grad(set_to_none=True)

                # 현재 모델로 행동을 예측한다.
                loss = loss_fn(model(x), y)

                # 각 가중치를 어떻게 바꾸면 손실이 줄어들지 계산한다.
                loss.backward()

                # 계산한 기울기를 이용해 AdamW가 가중치를 업데이트한다.
                opt.step()

                # loss는 이 배치의 평균 손실이다.
                # 마지막 배치는 작을 수 있으므로 샘플 수를 곱해 합산한다.
                batch_size = x.shape[0]
                train_loss_sum += loss.item() * batch_size
                train_count += batch_size

            # 전체 학습 샘플에 대한 평균 손실이다.
            train_loss = train_loss_sum / train_count

            # 2. 검증
            # 학습을 마친 현재 모델을 별도로 떼어둔 시연에서 평가한다.
            model.eval()

            val_loss_sum = 0.0
            val_count = 0

            # 검증에서는 기울기를 계산할 필요가 없다.
            # 가중치 업데이트 없이 예측 오차만 측정한다.
            with torch.no_grad():
                for x, y in dl_va:
                    prediction = model(x)
                    loss = loss_fn(prediction, y)

                    batch_size = x.shape[0]
                    val_loss_sum += loss.item() * batch_size
                    val_count += batch_size

            val_loss = val_loss_sum / val_count

            # 3. 가장 좋은 모델 저장
            # 검증 손실이 이전 최저값보다 작아졌을 때 체크포인트를 저장한다.
            # 따라서 ckpt.pt에는 검증 손실이 가장 낮았던 모델이 남는다.
            improved = val_loss < best

            if improved:
                best = val_loss

                save_checkpoint(
                    a.out / "ckpt.pt",
                    model,
                    config,
                    x_norm,
                    y_norm,
                    epoch=epoch,
                    val_loss=val_loss,
                )

            # 4. 학습 기록 저장
            # time_s는 학습 시작 후 지금까지 걸린 총 시간이다.
            elapsed = round(time.time() - t0, 1)

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                elapsed,
            ])

            # 현재까지의 기록을 파일에 반영한다.
            log.flush()

            # CMD 또는 PowerShell에서 진행 상황을 볼 수 있게 출력한다.
            # CSV의 epoch는 0부터, 화면 표시는 1부터 시작한다.
            status = " | best model saved" if improved else ""
            print(
                f"Epoch {epoch + 1}/{a.epochs}"
                f" | train_loss={train_loss:.6f}"
                f" | val_loss={val_loss:.6f}"
                f" | elapsed={elapsed:.1f}s"
                f"{status}",
                flush=True,
            )


if __name__ == '__main__':
    main()
