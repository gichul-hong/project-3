# 🚀 Subject-driven Customization (VERILUX Term Project)

> **베이스 모델**: `stabilityai/stable-diffusion-3.5-medium` (MMDiT Rectified Flow 2.5B)  
> **평가 모델**: `openai/clip-vit-base-patch32` & `openai/clip-vit-large-patch14`, `facebookresearch/dinov2`  
> **실행 환경**: Google Colab A100-SXM4-40GB GPU

---

## 📂 1. 파이썬 스크립트 및 핵심 파이프라인 안내

루트 디렉토리에 있는 각 파이썬(`.py`) 스크립트의 기능과 사용법입니다.

| 파일명 | 역할 및 설명 | 주요 사용 예시 |
| :--- | :--- | :--- |
| **`run_exp08_pipeline.py`** | **[마스터 파이프라인]** Class Prior 생성 $\rightarrow$ DreamBooth-LoRA 학습 $\rightarrow$ Controlled ODE Inversion 추론 $\rightarrow$ 다차원 정밀 평가 $\rightarrow$ Git/Drive 자동 동기화 일괄 수행 | `python run_exp08_pipeline.py` |
| **`train_dreambooth_sd3.py`** | **[학습]** Flow Matching Dual Loss ($\mathcal{L}_{inst} + \lambda_{prior} \mathcal{L}_{prior}$) 기반 True DreamBooth-LoRA 학습 (Rank 64, 1000 Steps) | `python train_dreambooth_sd3.py --concept all --prior_loss_weight 0.3` |
| **`generate_hybrid.py`** | **[하이브리드 추론]** 학습된 DreamBooth LoRA 가중치 + Heun 50-Step Controlled ODE Inversion 결합 추론 및 동적 서브젝트 라우팅 | `python generate_hybrid.py --concept all --subject_routing --scheduler heun --steps 50` |
| **`evaluate_extended.py`** | **[확장 평가기]** CLIP-L/14 + DINOv2-I + 4대 Taxonomy(화풍/속성/배경/동작) + Intra-cluster Diversity 다차원 측정 | `python evaluate_extended.py --exp_dir all --data_dir ./dataset` |
| **`update_all_reports.py`** | **[보고서 갱신]** 8개 전체 실험의 정량 지표와 파라미터를 읽어 마크다운 리포트 자동 생성 | `python update_all_reports.py` |
| **`generate_experiment_viewer.py`** | **[시각화]** 전체 8개 실험과 10개 서브젝트의 생성 이미지를 웹 브라우저에서 직접 비교하는 단일 HTML 뷰어 생성 | `python generate_experiment_viewer.py` |
| **`evaluation.py`** | **[공식 채점기]** CLIP-B/32 모델을 통한 Text-to-Image (CLIP-T) 및 Image-to-Image (CLIP-I) 표준 측정 | `python evaluation.py --dataset ./dataset --prompts ./prompt --concept <name> --images <dir>` |

---

## 📊 2. 실험(Experiment) 이력 및 성능 요약 (8단 비교)

| 실험 ID | 실험명 | 핵심 기법 | 공식 CLIP-T (↑) | 공식 CLIP-I (↑) | Total (T+I) | 결과 폴더 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp-01** | RF-Inversion Baseline | Controlled ODE Inversion (단일 Ref) | 0.2950 | **0.7831** | **1.0781** | [`experiments/01_rf_inversion_baseline/`](file:///content/project-3/experiments/01_rf_inversion_baseline/) |
| **Exp-03** | Augmented SD3.5 LoRA | 5종 증강 기반 MMDiT LoRA (Rank 16) | **0.3332** | 0.6645 | 0.9977 | [`experiments/03_lora_augmented/`](file:///content/project-3/experiments/03_lora_augmented/) |
| **Exp-04** | LoRA + RF-Inversion Hybrid | LoRA 가중치 + Controlled ODE 융합 | 0.3082 | 0.7634 | 1.0716 | [`experiments/04_lora_rf_hybrid/`](file:///content/project-3/experiments/04_lora_rf_hybrid/) |
| **Exp-05** | LoRA High-Quality | T5-XXL + Rank 64 (1,000 Steps) | 0.3239 | 0.6731 | 0.9970 | [`experiments/05_lora_hq/`](file:///content/project-3/experiments/05_lora_hq/) |
| **Exp-06** | Hybrid Adaptive $\eta$ Multi-ref | Multi-ref Latent Avg + Cosine Adaptive $\eta$ | 0.3241 | 0.7192 | 1.0433 | [`experiments/06_hybrid_adaptive/`](file:///content/project-3/experiments/06_hybrid_adaptive/) |
| **Exp-07** | Heun 50-Step Custom Neg | Heun 2차 ODE Solver (50 Steps) | 0.3224 | 0.7234 | 1.0458 | [`experiments/07_heun_custom_neg/`](file:///content/project-3/experiments/07_heun_custom_neg/) |
| **Exp-08** | **True DreamBooth Prior Loss** | **Dual Flow Loss ($\lambda_{prior}=0.3$) + Heun 50-Step** | 0.3273 | 0.6948 | 1.0221 | [`experiments/08_dreambooth_prior_loss/`](file:///content/project-3/experiments/08_dreambooth_prior_loss/) |
| **Exp-09** | **Subject Adaptive Routing** | **서브젝트 동적 라우팅 + 프롬프트 디테일 강화** | 0.3268 | 0.6908 | 1.0176 | [`experiments/09_subject_adaptive_routing/`](file:///content/project-3/experiments/09_subject_adaptive_routing/) |

> 📌 상세 실험 분석 및 서브젝트별 22개 CLIP 점수표: [docs/EXPERIMENT_HISTORY.md](file:///content/project-3/docs/EXPERIMENT_HISTORY.md)

---

## 📁 3. 주요 디렉토리 구조

```
/content/project-3/
├── 📁 data/class_priors/               ← 10개 클래스 generic prior 이미지 (400장)
├── 📁 augmentation/                    ← 5종 전처리 증강 데이터셋 (서브젝트당 20~75장 & 캡션)
├── 📁 checkpoints/
│   ├── exp05_lora_hq/                  ← High-Rank LoRA 가중치 (10종)
│   └── exp08_dreambooth_lora/          ← True DreamBooth-LoRA 가중치 (10종)
├── 📁 experiments/                     ← 실험 버전별 생성 이미지(100장) 및 리포트
│   ├── 01_rf_inversion_baseline/       ← [Exp-01 Inversion]
│   ├── 03_lora_augmented/             ← [Exp-03 LoRA R16]
│   ├── 04_lora_rf_hybrid/             ← [Exp-04 Hybrid Euler]
│   ├── 05_lora_hq/                    ← [Exp-05 LoRA R64 HQ]
│   ├── 06_hybrid_adaptive/            ← [Exp-06 Multi-ref Adaptive]
│   ├── 07_heun_custom_neg/            ← [Exp-07 Heun 50-Step]
│   ├── 08_dreambooth_prior_loss/      ← [Exp-08 Prior Loss]
│   └── 09_subject_adaptive_routing/   ← [Exp-09 Dynamic Routing]
├── 📁 docs/                            ← 개발 가이드, 증강 계획서, 실험 히스토리
├── 📁 dataset/                         ← 원본 레퍼런스 데이터셋
├── 📁 prompt/                          ← 서브젝트별 10개 테스트 프롬프트
├── 📄 experiment_viewer.html           ← 인터랙티브 결과 시각화 뷰어
└── 📄 README.md                        ← 전체 프로젝트 및 스크립트 가이드
```
