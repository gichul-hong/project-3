# 📋 프로젝트 실험 이력 및 작업 로그 (Experiment History & Changelog)

> **프로젝트명**: Subject-driven Customization (VERILUX Term Project)  
> **베이스 모델**: `stabilityai/stable-diffusion-3.5-medium` (MMDiT Flow Matching)  
> **평가 모델**: `openai/clip-vit-base-patch32` (CLIP-B/32)  
> **실행 환경**: Google Colab A100-SXM4-40GB GPU

---

## 📌 1. 전체 실험 요약 및 성능 비교표

| 실험 ID | 실험명 (Experiment) | 데이터셋 | 방법론 (Method) | 주요 하이퍼파라미터 | 평균 CLIP-T (↑) | 평균 CLIP-I (↑) | Total (T+I) | 산출물 위치 |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp-01** | RF-Inversion Baseline | `./dataset` | Controlled ODE Inversion | Steps=28, CFG=7.0, $\tau$=0.7, $\eta$=0.9 | **0.2950** | **0.7831** | **1.0781** | [experiments/01_rf_inversion_baseline/](file:///content/project-3/experiments/01_rf_inversion_baseline/) |
| **Exp-03** | Augmented SD3.5 LoRA | `./augmentation` | 증강 데이터 기반 MMDiT LoRA | Rank=16, $\alpha$=32, LR=1e-4, Steps=200 | **0.3332** | **0.6645** | **0.9977** | [experiments/03_lora_augmented/](file:///content/project-3/experiments/03_lora_augmented/) |
| **Exp-04** | LoRA + RF-Inversion Hybrid | `./augmentation` | LoRA 가중치 + Controlled ODE 결합 | LoRA + $\tau$=0.7, $\eta$=0.9 | *예정 (아이디어)* | *예정* | *예정* | `./experiments/04_lora_rf_hybrid/` |

---

## 📊 2. 서브젝트별 상세 정량 평가 비교 (Exp-01 vs Exp-03)

| 서브젝트 (Concept) | Exp-01 Inversion (CLIP-T) | Exp-03 LoRA (CLIP-T) | Exp-01 Inversion (CLIP-I) | Exp-03 LoRA (CLIP-I) | 비고 / 분석 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `actionfigure_2` | 0.2755 | **0.3224** *(+0.0469)* | **0.6950** | 0.4622 | 텍스트 반영력 대폭 향상 |
| `decoritems_woodenpot` | 0.3151 | **0.3563** *(+0.0412)* | **0.7575** | 0.6340 | 테이블/꽃/성 배경 완벽 합성 |
| `furniture_sofa2` | 0.2629 | **0.3248** *(+0.0619)* | **0.9221** | 0.7678 | 수영장/해변 등 다양한 배경 적응 |
| `instrument_music2` | 0.2912 | **0.3507** *(+0.0595)* | **0.8181** | 0.7244 | 네온/사이버펑크 구도 우수 |
| `luggage_backpack1` | 0.3141 | **0.3389** *(+0.0248)* | **0.8628** | 0.7322 | 숲길/카페 구도 준수 |
| `person_3` | 0.2813 | **0.3115** *(+0.0302)* | **0.6335** | 0.5163 | 우주복/소방관 등 복장 변경 우수 |
| `pet_cat5` | 0.2928 | **0.3287** *(+0.0359)* | **0.8578** | 0.7944 | 선글라스/헤드폰 착용 완벽 생성 |
| `scene_waterfall` | 0.3102 | **0.3438** *(+0.0336)* | **0.8171** | 0.7514 | 겨울 설경/사이버펑크 조화 |
| `transport_tank` | 0.2999 | **0.3341** *(+0.0342)* | **0.6493** | 0.5599 | 화성/우주비행사 배경 반영 |
| `wearable_jacket1` | 0.3069 | **0.3208** *(+0.0139)* | **0.8179** | 0.7021 | 마네킹/로봇 착용 합성 |
| **전체 평균 (TOTAL AVG)** | **0.2950** | **0.3332** *(+0.0382)* | **0.7831** | **0.6645** | **CLIP-T 0.33 돌파 달성** |

> 💡 **핵심 인사이트**:  
> - **LoRA 파인튜닝 (Exp-03)**은 모든 10개 서브젝트에서 **Text-to-Image (CLIP-T) 점수를 평균 +0.0382 (최대 +0.0619) 대폭 향상**시켰습니다. 새로운 배경과 상황 프롬프트를 훨씬 더 자연스럽게 생성합니다.  
> - **RF-Inversion (Exp-01)**은 원본 레퍼런스 Latent를 보정하므로 **Image-to-Image (CLIP-I, 0.7831)**에서 높은 보존력을 보였습니다.  
> - **최종 목표 (Exp-04 아이디어)**: LoRA 가중치로 학습된 모델 위에 RF-Inversion을 결합하면 CLIP-T(0.33+)와 CLIP-I(0.80+)를 동시에 극대화할 수 있습니다!

---

## 🛠️ 3. 구축된 전체 파일 및 디렉토리 구조

```
/content/project-3/
├── 📁 augmentation/                    ← 5종 전처리 증강 데이터셋 (서브젝트당 20~75장 & 캡션)
├── 📁 checkpoints/                     ← 학습된 서브젝트별 LoRA 가중치 (.safetensors)
│   ├── lora_actionfigure_2/
│   ├── lora_decoritems_woodenpot/
│   └── ... (총 10개)
│
├── 📁 experiments/
│   ├── 01_rf_inversion_baseline/       ← [Exp-01] 100장 이미지 + EVALUATION_REPORT.md
│   └── 03_lora_augmented/             ← [Exp-03] 100장 이미지 + EVALUATION_REPORT.md
│
├── 📁 docs/
│   ├── AUGMENTATION_PLAN.md            ← 증강 기법 설계 문서
│   ├── DEVELOPMENT_GUIDE.md           ← 종합 개발 가이드
│   └── EXPERIMENT_HISTORY.md          ← [현재 문서] 전체 실험 이력 및 정량 비교
│
├── 📄 train_lora_sd3.py               ← MMDiT LoRA 파인튜닝 파이프라인
├── 📄 generate_lora.py                ← LoRA 추론 & 자동 평가 스크립트
├── 📄 generate_inversion.py           ← Controlled ODE RF-Inversion 스크립트
├── 📄 evaluation.py                   ← 공식 CLIP-B/32 채점 스크립트
└── 📄 dataset_viewer.html             ← 데이터셋 시각화 뷰어
```
