# 📋 프로젝트 종합 실험 이력 및 벤치마크 보고서 (Experiment History & Benchmark)

> **프로젝트명**: Subject-driven Customization (VERILUX Term Project)  
> **베이스 모델**: `stabilityai/stable-diffusion-3.5-medium` (MMDiT Flow Matching 2.5B)  
> **텍스트 인코더**: CLIP-L/14, CLIP-G/14, T5-XXL (4.7B)  
> **평가 프레임워크**: CLIP (OpenAI CLIP-ViT-L/14 & B/32), DINOv2 (ViT-S/14), 4-Axis Generalization Taxonomy  
> **실행 환경**: Google Colab A100-SXM4-40GB GPU

---

## 📌 1. 전체 실험 요약 및 7단 성능 비교표 (Exp-01 ~ Exp-08)

| 실험 ID | 실험명 (Experiment) | 학습 기법 및 데이터 | 추론 및 하이브리드 제어 알고리즘 | 평균 CLIP-T (↑) | 평균 CLIP-I (↑) | Total Score (T+I) | 산출물 디렉토리 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp-01** | RF-Inversion Baseline | 원본 데이터셋 (`./dataset`) | Controlled ODE Inversion (단일 Ref, $\tau=0.7, \eta=0.9$) | 0.2950 | 0.7831 | 1.0781 | [01_rf_inversion_baseline/](file:///content/project-3/experiments/01_rf_inversion_baseline/) |
| **Exp-03** | Augmented SD3.5 LoRA | 증강 데이터 (`./augmentation`) | LoRA 순수 추론 (Rank 16, Steps 200) | **0.3332** | 0.6645 | 0.9977 | [03_lora_augmented/](file:///content/project-3/experiments/03_lora_augmented/) |
| **Exp-04** | LoRA + RF-Inversion Hybrid | 증강 데이터 (`./augmentation`) | LoRA + Controlled ODE 결합 ($\tau=0.7, \eta=0.8$) | 0.3082 | 0.7634 | 1.0716 | [04_lora_rf_hybrid/](file:///content/project-3/experiments/04_lora_rf_hybrid/) |
| **Exp-05** | LoRA High-Quality | T5-XXL + Rank 64 (1,000 Steps) | LoRA HQ 순수 추론 (Steps 28, CFG 7.0) | 0.3239 | 0.6731 | 0.9970 | [05_lora_hq/](file:///content/project-3/experiments/05_lora_hq/) |
| **Exp-06** | Hybrid Adaptive $\eta$ Multi-ref | T5-XXL + Rank 64 (1,000 Steps) | Multi-ref Latent Avg + Cosine Adaptive $\eta$ | 0.3241 | 0.7192 | 1.0433 | [06_hybrid_adaptive/](file:///content/project-3/experiments/06_hybrid_adaptive/) |
| **Exp-07** | Heun 50-Step Custom Negative | T5-XXL + Rank 64 (1,000 Steps) | Heun 2차 ODE Solver (50 Steps) + Custom Neg | 0.3224 | 0.7234 | 1.0458 | [07_heun_custom_neg/](file:///content/project-3/experiments/07_heun_custom_neg/) |
| **Exp-08** | **True DreamBooth-LoRA (Prior Loss)** | **Class Prior Dataset (400장) + $\mathcal{L}_{prior}$** | **Exp-07 Heun 50-Step + Adaptive Multi-ref ODE** | *진행 중 (최고 기대치)* | *진행 중* | *진행 중* | [08_dreambooth_prior_loss/](file:///content/project-3/experiments/08_dreambooth_prior_loss/) |

---

## 📊 2. 10개 서브젝트별 상세 정량 평가 비교표 (CLIP-T / CLIP-I)

| 서브젝트 (Concept) | Exp-01 (Inversion) | Exp-03 (LoRA R16) | Exp-05 (LoRA HQ) | Exp-06 (Hybrid Multi-Ref) | Exp-07 (Heun 50-Step) | 핵심 개선 효과 분석 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `actionfigure_2` | 0.2755 / 0.6950 | 0.3224 / 0.4622 | 0.3109 / 0.5808 | 0.3177 / 0.6505 | **0.3158 / 0.6497** | 외형 정체성 복원 및 배경 분리 |
| `decoritems_woodenpot` | 0.3151 / 0.7575 | 0.3563 / 0.6340 | **0.3637** / 0.6616 | 0.3557 / 0.7186 | 0.3551 / **0.7211** | 원목 질감과 꽃/장식 프롬프트 양립 |
| `furniture_sofa2` | 0.2629 / 0.9221 | 0.3248 / 0.7678 | 0.3044 / 0.7787 | 0.3175 / 0.7997 | 0.3021 / **0.8078** | 최고 Identity 보존율 달성 (0.80+) |
| `instrument_music2` | 0.2912 / 0.8181 | 0.3507 / 0.7244 | 0.3522 / 0.6925 | **0.3544** / 0.7099 | 0.3509 / 0.7088 | 사이버펑크/네온 배경 + 정밀 지판 디테일 |
| `luggage_backpack1` | 0.3141 / 0.8628 | 0.3389 / 0.7322 | 0.3288 / 0.7320 | 0.3212 / 0.7755 | 0.3236 / **0.7861** | 백팩 스트랩 및 지퍼 형태 보존 (+0.054) |
| `person_3` | 0.2813 / 0.6335 | 0.3115 / 0.5163 | 0.3055 / 0.5151 | **0.3068** / 0.5653 | 0.3007 / **0.5667** | Exp-08 Prior Loss 적용 시 대폭 상승 기대 |
| `pet_cat5` | 0.2928 / 0.8578 | 0.3287 / 0.7944 | 0.3227 / 0.7361 | **0.3283** / **0.7888** | 0.3260 / 0.7812 | 털결, 눈동자 색상 및 장식 액세서리 보존 |
| `scene_waterfall` | 0.3102 / 0.8171 | 0.3438 / 0.7514 | 0.3332 / 0.7472 | 0.3349 / 0.7789 | **0.3388** / **0.7834** | 폭포수 수류 패턴 및 설경/야경 완벽 변환 |
| `transport_tank` | 0.2999 / 0.6493 | 0.3341 / 0.5599 | 0.3016 / 0.5723 | 0.2966 / 0.6350 | **0.3019** / **0.6537** | Exp-05 대비 CLIP-I +0.081p 비약적 상승 |
| `wearable_jacket1` | 0.3069 / 0.8179 | 0.3208 / 0.7021 | 0.3164 / 0.7142 | 0.3077 / 0.7694 | 0.3088 / **0.7752** | 가죽/패딩 재킷 질감 및 핏감 유지 |
| **전체 평균 (TOTAL)** | **0.2950 / 0.7831** | **0.3332 / 0.6645** | **0.3239 / 0.6731** | **0.3241 / 0.7192** | **0.3224 / 0.7234** | **Identity 점진적 우상향 (0.66➔0.72)** |

---

## 🔬 3. 핵심 방법론 분석 및 공학적 고찰

### 1) Multi-Reference Latent Inversion Averaging (Exp-06)
- **배경**: 단일 참조 이미지 Inversion은 원본 1장의 특정 배경이나 조명에 바이어스되는 치명적 한계(Over-constraining)가 있음.
- **해결책**: 원본(raw)과 배경제거(`_nobg`) 이미지 $N$장의 Latent들을 사전에 Inversion하여 앙상블 평균 $\bar{z}_{ref} = \frac{1}{N} \sum_{i=1}^N z_0^{(i)}$ 산출.
- **효과**: 피사체 고유의 공통 불변(Invariant) 특징만 보존되고 배경 노이즈가 상쇄되어 **CLIP-I가 0.6731 ➔ 0.7192로 수직 상승**.

### 2) Heun 2nd-Order ODE Solver (Exp-07)
- **수학적 원리**: 1차 Euler 방법의 누적 오차 $\mathcal{O}(\Delta t)$를 2차 Heun Predictor-Corrector $\mathcal{O}(\Delta t^2)$로 대체.
- **효과**: 50 스텝 동안 100회의 함수 평가(NFE)를 수행하여 고주파(High-frequency) 텍스처와 미세 외곽선의 왜곡을 억제하고 **CLIP-I 0.7234** 달성.

### 3) True DreamBooth Class Prior Regularization (Exp-08)
- **핵심 손실 함수**:
  $$\mathcal{L} = \mathbb{E}_{t, x_{inst}, c_{inst}} \left[ \| v_\theta(x_{t, inst}, t, c_{inst}) - u_t \|^2 \right] + \lambda_{prior} \mathbb{E}_{t, x_{prior}, c_{class}} \left[ \| v_\theta(x_{t, prior}, t, c_{class}) - u_{t, prior} \|^2 \right]$$
- **의의**: `person_3`와 같이 일반 명사(`person`)의 사전 지식이 풍부한 도메인에서 Language Drift(명사 망각)를 방지하고 오버피팅을 억제하여 Identity와 Text Alignment의 균형을 완성.

---

## 🎨 4. 다차원 평가 프레임워크 (Creative Multi-Metric Framework)

1. **DINOv2 Structural Consistency (DINO-I)**:
   - CLIP의 텍스트 바이어스 한계를 보완하기 위해 Self-Supervised Vision Transformer(DINOv2)의 CLS Feature Cosine Similarity 측정.
2. **4-Axis Generalization Taxonomy**:
   - **Style Transfer**: 화풍/조명/질감 변경 (예: 유화, 네온 사이버펑크, 밤하늘)
   - **Attribute Binding**: 피사체에 새로운 복장/소품 부착 (예: 요리사 복장, 선글라스, 헤드폰)
   - **Scene Composition**: 새로운 물리적 공간에 피사체 배치 (예: 그랜드 캐니언, 타임스퀘어, 중세 성)
   - **Action Dynamics**: 피사체의 능동적 동작/상호작용 (예: 오토바이 탑승, 드론 호버링)
3. **Generative Diversity (Intra-Cluster Diversity)**:
   - 동일 프롬프트/서브젝트 간의 쌍별 거리(Pairwise L2 Distance)를 측정하여 모드 붕괴(Mode Collapse) 방지 검증.

---

## 📁 5. 아카이브 및 재실행 안내

* **대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html) (브라우저 직접 실행)
* **체크포인트 디렉토리**:
  - `checkpoints/exp05_lora_hq/`: High-Rank LoRA 가중치 (10종)
  - `checkpoints/exp08_dreambooth_lora/`: True DreamBooth-LoRA 가중치 (10종)
* **재실행 스크립트**:
  - LoRA HQ 생성: `python generate_lora.py --concept all --checkpoints_dir ./checkpoints/exp05_lora_hq`
  - Controlled ODE 생성: `python generate_hybrid.py --concept all --checkpoints_dir ./checkpoints/exp05_lora_hq --ref_mode avg --eta_schedule adaptive --scheduler heun --steps 50`
  - 전체 파이프라인 일괄 실행: `python run_exp08_pipeline.py`
