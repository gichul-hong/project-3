# 📊 Exp-08: True DreamBooth-LoRA (Prior Loss lambda=0.3) + Null-Text Heun ODE

- **실험 디렉토리**: `08_dreambooth_prior_loss`
- **방법론 (Method)**: `SD3.5 True DreamBooth-LoRA (Dual Flow Loss lambda=0.3) + Null-Text Controlled ODE Inversion`
- **주요 파라미터 (Hyperparameters)**: `DreamBooth-LoRA Rank 64, lambda_prior=0.3 + Null-Text Inversion + Heun 50 Steps`
- **전체 평균 결과**: **CLIP-T: 0.3273 | CLIP-I: 0.6948 | Combined Total: 1.0221**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

SD3.5 Base 생성 Class Prior 정규화 데이터셋과 Dual Flow Loss(lambda=0.3)를 통해 Language Drift를 억제하고, Null-text Inversion으로 순수 기하학적 잠재 궤적 역추적.

### 주요 기법:
* **방법론 상세**: SD3.5 True DreamBooth-LoRA (Dual Flow Loss lambda=0.3) + Null-Text Controlled ODE Inversion
* **파라미터 구성**: `DreamBooth-LoRA Rank 64, lambda_prior=0.3 + Null-Text Inversion + Heun 50 Steps`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3396 | 0.5460 | 0.8856 |
| `decoritems_woodenpot` | 0.3649 | 0.6634 | 1.0283 |
| `furniture_sofa2` | 0.2986 | 0.7952 | 1.0938 |
| `instrument_music2` | 0.3545 | 0.6981 | 1.0526 |
| `luggage_backpack1` | 0.3300 | 0.7820 | 1.1120 |
| `person_3` | 0.2993 | 0.5550 | 0.8543 |
| `pet_cat5` | 0.3240 | 0.7842 | 1.1082 |
| `scene_waterfall` | 0.3509 | 0.7870 | 1.1379 |
| `transport_tank` | 0.2978 | 0.5762 | 0.8740 |
| `wearable_jacket1` | 0.3131 | 0.7613 | 1.0744 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3273** | **0.6948** | **1.0221** |

---

## 💡 3. 심층 결과 분석 및 고찰

LoRA의 한정된 용량 내에서 언어 망각을 방지하면서 Subject Identity를 극대화하고, Null-Text Inversion을 통해 프롬프트 자유도와 정체성의 최적 조화를 달성.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/08_dreambooth_prior_loss/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
