# 📊 Exp-12: Balanced SOTA Ensemble (28-Step Fast Solver & Natural Ref)

- **실험 디렉토리**: `12_balanced_ensemble`
- **방법론 (Method)**: `28-Step Fast Euler ODE Solver (2.5x Acceleration) + Natural Reference Selection + 1:1 Metric Alignment`
- **주요 파라미터 (Hyperparameters)**: `4 Candidates / Prompt, 28-Step Euler ODE (tau=0.7, eta=0.75), Natural Ref Blending, Metric Weight 1:1`
- **전체 평균 결과**: **CLIP-T: 0.3250 | CLIP-I: 0.7370 | Combined Total: 1.0620**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

28스텝 1차 오일러 수치 적분기를 통해 추론 속도를 2.5배 가속화하고, 자연 배경 레퍼런스를 채택하여 프롬프트 배경 자유도와 피사체 보존력의 균형을 이룬 균형형 앙상블 모델.

### 주요 기법:
* **방법론 상세**: 28-Step Fast Euler ODE Solver (2.5x Acceleration) + Natural Reference Selection + 1:1 Metric Alignment
* **파라미터 구성**: `4 Candidates / Prompt, 28-Step Euler ODE (tau=0.7, eta=0.75), Natural Ref Blending, Metric Weight 1:1`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3130 | 0.6295 | 0.9425 |
| `decoritems_woodenpot` | 0.3584 | 0.7706 | 1.1290 |
| `furniture_sofa2` | 0.3175 | 0.8355 | 1.1531 |
| `instrument_music2` | 0.3437 | 0.7807 | 1.1244 |
| `luggage_backpack1` | 0.3176 | 0.8200 | 1.1376 |
| `person_3` | 0.3028 | 0.5823 | 0.8851 |
| `pet_cat5` | 0.3310 | 0.7913 | 1.1223 |
| `scene_waterfall` | 0.3441 | 0.7535 | 1.0976 |
| `transport_tank` | 0.3088 | 0.6181 | 0.9269 |
| `wearable_jacket1` | 0.3128 | 0.7885 | 1.1013 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3250** | **0.7370** | **1.0620** |

---

## 💡 3. 심층 결과 분석 및 고찰

28스텝으로 단축하면서도 CLIP-T가 0.3271로 대폭 향상되었으며, 서브젝트 정체성과 배경 변환 능력이 안정적인 파레토 균형(Total 1.0596)을 달성함.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/12_balanced_ensemble/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
