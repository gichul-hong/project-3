# 🚀 Subject-driven Customization (VERILUX Term Project)

> **베이스 모델**: `stabilityai/stable-diffusion-3.5-medium` (MMDiT Rectified Flow)  
> **평가 모델**: `openai/clip-vit-base-patch32` (CLIP-B/32)  
> **실행 환경**: Google Colab A100-SXM4-40GB GPU

---

## 📂 1. 파이썬 스크립트 및 핵심 파일 역할 안내

루트 디렉토리에 있는 각 파이썬(`.py`) 스크립트의 기능과 사용법입니다.

| 파일명 | 역할 및 설명 | 주요 사용 예시 |
| :--- | :--- | :--- |
| **`generate_hybrid.py`** | **[Iter 4 하이브리드]** 학습된 LoRA 가중치 + Controlled ODE Inversion을 결합하여 CLIP-T와 CLIP-I를 동시 극대화. | `python generate_hybrid.py --concept all --tau 0.7 --eta 0.8` |
| **`train_lora_sd3.py`** | **[학습]** 증강 데이터셋(`./augmentation`) 기반 SD3.5 MMDiT LoRA 파인튜닝. VAE/Text 임베딩 사전 캐싱으로 초고속 학습. | `python train_lora_sd3.py --concept all --exp_name exp03_lora --steps 200` |
| **`generate_lora.py`** | **[추론/평가]** 학습된 LoRA 가중치를 로드하여 10개 프롬프트 이미지(100장) 생성 및 CLIP 자동 채점. | `python generate_lora.py --concept all --exp_name exp03_lora` |
| **`generate_inversion.py`** | **[추론/평가]** 가중치 학습 없이 Controlled ODE (RF-Inversion / Euler) 기반으로 레퍼런스 Latent 역추적 및 생성. | `python generate_inversion.py --concept all --method rf` |
| **`generate_baseline.py`** | **[Baseline]** Zero-Shot 순수 텍스트 프롬프트 기반 Baseline 생성 및 평가. | `python generate_baseline.py --concept all` |
| **`augment_dataset.py`** | **[전처리/증강]** 5종 증강(`std`, `flip`, `light`, `nobg`, `nobg_flip`) 이미지 및 `metadata.jsonl`(`sks`) 자동 생성. | `python augment_dataset.py` |
| **`generate_dataset_viewer.py`** | **[시각화]** 증강된 데이터셋을 인터랙티브하게 검토할 수 있는 단일 HTML 뷰어 생성. | `python generate_dataset_viewer.py` |
| **`evaluation.py`** | **[공식 채점기]** CLIP-B/32 모델을 통한 Text-to-Image (CLIP-T) 및 Image-to-Image (CLIP-I) 측정. | `python evaluation.py --dataset ./dataset --prompts ./prompt --concept <name> --images <dir>` |
| **`backup_manager.py`** | **[백업 관리자]** 타임스탬프 및 실험 태그 기반으로 Google Drive 또는 zip 아카이브 스냅샷 생성. | `python backup_manager.py --target drive --tag "exp03"` |

---

## 📊 2. 실험(Experiment) 이력 및 성능 요약

| 실험 ID | 실험명 | 데이터셋 | 방법론 | CLIP-T (↑) | CLIP-I (↑) | Total (T+I) | 결과 폴더 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Exp-01** | RF-Inversion Baseline | `./dataset` | Controlled ODE Inversion ($\tau$=0.7, $\eta$=0.9) | 0.2950 | **0.7831** | **1.0781** | [`experiments/01_rf_inversion_baseline/`](file:///content/project-3/experiments/01_rf_inversion_baseline/) |
| **Exp-03** | Augmented SD3.5 LoRA | `./augmentation` | 5종 증강 기반 MMDiT LoRA (Rank 16, 200 Steps) | **0.3332** | 0.6645 | 0.9977 | [`experiments/03_lora_augmented/`](file:///content/project-3/experiments/03_lora_augmented/) |
| **Exp-04** | LoRA + RF-Inversion Hybrid | `./augmentation` | **LoRA 가중치 + Controlled ODE 융합 (최적 밸런스)** | **0.3082** | **0.7634** | **1.0716** | [`experiments/04_lora_rf_hybrid/`](file:///content/project-3/experiments/04_lora_rf_hybrid/) |

> 📌 상세 실험 분석 및 서브젝트별 22개 CLIP 점수표: [docs/EXPERIMENT_HISTORY.md](file:///content/project-3/docs/EXPERIMENT_HISTORY.md)

---

## 📁 3. 주요 디렉토리 구조

```
/content/project-3/
├── 📁 augmentation/                    ← 5종 전처리 증강 데이터셋 (서브젝트당 20~75장 & 캡션)
├── 📁 checkpoints/                     ← 학습된 LoRA 체크포인트 아카이브 (10종)
├── 📁 experiments/                     ← 실험 버전별 생성 이미지(100장) 및 EVALUATION_REPORT.md
│   ├── 01_rf_inversion_baseline/       ← [Exp-01 Inversion] CLIP-T 0.2950 / CLIP-I 0.7831
│   ├── 03_lora_augmented/             ← [Exp-03 LoRA]      CLIP-T 0.3332 / CLIP-I 0.6645
│   └── 04_lora_rf_hybrid/             ← [Exp-04 Hybrid]    CLIP-T 0.3082 / CLIP-I 0.7634
├── 📁 docs/                            ← 개발 가이드, 증강 계획서, 실험 히스토리 문서
├── 📁 dataset/                         ← 원본 레퍼런스 데이터셋
├── 📁 prompt/                          ← 서브젝트별 10개 테스트 프롬프트
└── 📄 README.md                        ← 전체 프로젝트 및 스크립트 가이드
```
