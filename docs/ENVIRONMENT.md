# 실행 환경과 명령어

Windows / Anaconda에서 사용한 프로젝트입니다. 기존 환경 이름은 `deeplearning`입니다. 아래 명세는 새 환경 구성을 위한 것이며, 당시 설치 환경 전체를 추출한 lock 파일은 아닙니다.

| 구성 | 명세 |
|---|---|
| Python | 3.12 · 설치·실행 미검증 |
| MuJoCo | 3.3.1 |
| robosuite | 1.5.2 |
| NumPy | 2 미만 |
| PyTorch | 원본 requirements에서 버전 미고정 |
| 기타 | opencv-python-headless, imageio-ffmpeg, pynput, matplotlib |
| 장치 | 제공된 BC·PPO 스크립트는 CPU 텐서로 동작 |

## Python 3.12 환경 생성부터 시작

Anaconda Prompt에서 이 폴더로 이동한 다음 실행합니다.

```bat
cd /d "<RopeT_Portfolio 폴더 경로>"
conda create -n ropet python=3.12 -y
conda activate ropet
python -m pip install -r requirements.txt
```

새 환경이 필요하면 활성화·설치 대신 다음을 사용합니다.

```bat
conda env create -f environment.yml
conda activate ropet
```

## 포함된 가중치로 평가

```bat
python il/eval_policy.py --agent bc --ckpt checkpoints/obs2_bc.pt --n 50 --steps 600 --out runs/eval_obs2_local
```

화면으로 1회 확인:

```bat
python il/eval_policy.py --agent bc --ckpt checkpoints/obs2_bc.pt --n 1 --steps 600 --view --out runs/preview_obs2
```

## BC 재학습

```bat
python il/check_il.py --data demos/T03_selected_n73
python il/train_bc.py --data demos/T03_selected_n73 --obs-horizon 2 --act-horizon 1 --seed 0 --out runs/obs2_retrain
```

## PPO 추가 실험

```bat
python il/train_rl.py --bc-ckpt checkpoints/obs2_bc.pt --out runs/obs2_ppo_new
```

PPO 출력 폴더는 새 경로여야 합니다. 최종 선택 모델은 PPO가 아닌 BC입니다.

## 검증 범위

정리 과정에서는 문법과 파일 구성만 확인했습니다. 환경 설치, 모듈 import, MuJoCo 실행, 학습과 평가는 수행하지 않았습니다. 포함된 가중치는 로컬 `runs/final_version/!!obs2/ckpt.pt`의 사본입니다. 과거 팀 평가에 사용된 파일과 바이트 단위로 같은지는 확인되지 않았으므로 동일 성능을 보장하지 않습니다.
