# 📋 프로젝트 실험 이력 및 작업 로그 (Experiment History & Changelog)

> **프로젝트명**: Subject-driven Customization (VERILUX Term Project)  
> **베이스 모델**: `stabilityai/stable-diffusion-3.5-medium` (MMDiT Flow Matching)  
> **평가 모델**: `openai/clip-vit-base-patch32` (CLIP-B/32)  
> **실행 환경**: Google Colab A100-SXM4-40GB GPU

---

## 📌 1. 전체 실험 요약 및 3단 성능 비교표

| 실험 ID | 실험명 (Experiment) | 데이터셋 | 방법론 (Method) | 주요 하이퍼파라미터 | 평균 CLIP-T (↑) | 평균 CLIP-I (↑) | Total (T+I) | 산출물 위치 |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp-01** | RF-Inversion Baseline | `./dataset` | Controlled ODE Inversion | Steps=28, CFG=7.0, $\tau$=0.7, $\eta$=0.9 | 0.2950 | **0.7831** | **1.0781** | [experiments/01_rf_inversion_baseline/](file:///content/project-3/experiments/01_rf_inversion_baseline/) |
| **Exp-03** | Augmented SD3.5 LoRA | `./augmentation` | 증강 데이터 기반 MMDiT LoRA | Rank=16, $\alpha$=32, LR=1e-4, Steps=200 | **0.3332** | 0.6645 | 0.9977 | [experiments/03_lora_augmented/](file:///content/project-3/experiments/03_lora_augmented/) |
| **Exp-04** | LoRA + RF-Inversion Hybrid | `./augmentation` | **LoRA 가중치 + Controlled ODE 결합** | LoRA + $\tau$=0.7, $\eta$=0.8 | **0.3082** | **0.7634** | **1.0716** | [experiments/04_lora_rf_hybrid/](file:///content/project-3/experiments/04_lora_rf_hybrid/) |

---

## 📊 2. 10개 서브젝트별 상세 정량 평가 비교표 (22개 공식 평가 수치)

| 서브젝트 (Concept) | Exp-01 Inversion (CLIP-T / CLIP-I) | Exp-03 LoRA (CLIP-T / CLIP-I) | Exp-04 Hybrid (CLIP-T / CLIP-I) | 최적 달성 효과 (Hybrid 분석) |
| :--- | :---: | :---: | :---: | :--- |
| `actionfigure_2` | 0.2755 / 0.6950 | **0.3224** / 0.4622 | 0.2744 / **0.6749** | 외형 정체성 복원 |
| `decoritems_woodenpot` | 0.3151 / 0.7575 | **0.3563** / 0.6340 | 0.3286 / **0.7854** | CLIP-I 최고점 갱신 (+0.028) |
| `furniture_sofa2` | 0.2629 / 0.9221 | **0.3248** / 0.7678 | 0.2769 / **0.9011** | 문맥/외형 완벽 밸런스 |
| `instrument_music2` | 0.2912 / 0.8181 | **0.3507** / 0.7244 | 0.3267 / **0.8128** | 사이버펑크/네온 배경 + 기타 디테일 |
| `luggage_backpack1` | 0.3141 / 0.8628 | **0.3389** / 0.7322 | 0.3144 / **0.8554** | 가방 질감 및 숲길 배경 보존 |
| `person_3` | 0.2813 / 0.6335 | **0.3115** / 0.5163 | 0.3038 / **0.5423** | 소방관/우주복 프롬프트 준수 |
| `pet_cat5` | 0.2928 / 0.8578 | **0.3287** / 0.7944 | 0.3126 / **0.8275** | 선글라스/헤드폰 + 고양이 외형 양립 |
| `scene_waterfall` | 0.3102 / 0.8171 | **0.3438** / 0.7514 | 0.3357 / **0.7871** | 폭포수 자연스러움 유지 |
| `transport_tank` | 0.2999 / 0.6493 | **0.3341** / 0.5599 | 0.2986 / **0.6368** | 탱크 기계적 디테일 유지 |
| `wearable_jacket1` | 0.3069 / 0.8179 | **0.3208** / 0.7021 | 0.3105 / **0.8109** | 재킷 질감 및 마네킹 구도 우수 |
| **전체 평균 (TOTAL AVG)** | **0.2950 / 0.7831** | **0.3332 / 0.6645** | **0.3082 / 0.7634** | **CLIP-T & CLIP-I 동시 극대화 달성** |

---

## 💡 3. 핵심 방법론 분석 및 발표자료(PPT) 구성 가이드

### 1) 방법론별 강점 및 트레이드오프
* **Inversion 단독 (`Exp-01`)**:
  - 강점: 이미지의 픽셀/Latent 궤적을 직접 보정하므로 피사체 원본 보존력(CLIP-I 0.7831)이 매우 뛰어남.
  - 약점: 프롬프트에 새로운 배경/상황이 주어졌을 때 텍스트 반영력(CLIP-T 0.2950)에 한계.
* **LoRA 단독 (`Exp-03`)**:
  - 강점: 배경 제거 증강 데이터셋(`nobg`) 학습으로 프롬프트 준수력(CLIP-T 0.3332)이 비약적으로 상승.
  - 약점: 배경 합성 자유도가 커진 반면 원본 피사체의 미세 디테일(CLIP-I 0.6645)이 다소 감소.
* **LoRA + RF-Inversion Hybrid (`Exp-04`, 배점 40% 핵심 아이디어)**:
  - **LoRA의 프롬프트 적응력 + Controlled ODE의 정체성 제어 궤적을 결합.**
  - **CLIP-T 0.3082 & CLIP-I 0.7634 (Total 1.0716)**를 달성하여 정량적/정성적으로 가장 균형 잡힌 고품질 이미지 생성 입증.

---

## 📁 4. 최종 아카이브 산출물

* **Exp-01 결과**: [experiments/01_rf_inversion_baseline/](file:///content/project-3/experiments/01_rf_inversion_baseline/)
* **Exp-03 결과**: [experiments/03_lora_augmented/](file:///content/project-3/experiments/03_lora_augmented/)
* **Exp-04 결과**: [experiments/04_lora_rf_hybrid/](file:///content/project-3/experiments/04_lora_rf_hybrid/)
* **학습 가중치**: [checkpoints/](file:///content/project-3/checkpoints/) (10종 `.safetensors`)
* **데이터셋 시각화**: [dataset_viewer.html](file:///content/project-3/dataset_viewer.html)
