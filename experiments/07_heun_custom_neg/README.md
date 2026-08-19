# 📊 Exp-07: Controlled ODE Heun 50-Step Solver + Custom Negative Prompt

- **실험 디렉토리**: `07_heun_custom_neg`
- **방법론 (Method)**: `LoRA HQ + Heun 2nd-Order ODE Solver (50 Steps) + Subject-Specific Negative Prompts`
- **주요 파라미터 (Hyperparameters)**: `LoRA Rank 64 + Heun 50 Steps (100 NFE) + tau=0.7, eta=0.85 + Custom Negative Prompts`
- **전체 평균 결과**: **CLIP-T: 0.3224 | CLIP-I: 0.7234 | Combined Total: 1.0458**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

2차 정확도 Heun ODE Solver를 적용하여 50 스텝(100 NFE) 동안 미세 구조 왜곡을 억제하고, 서브젝트별 특화 Negative Prompt 적용.

### 주요 기법:
* **방법론 상세**: LoRA HQ + Heun 2nd-Order ODE Solver (50 Steps) + Subject-Specific Negative Prompts
* **파라미터 구성**: `LoRA Rank 64 + Heun 50 Steps (100 NFE) + tau=0.7, eta=0.85 + Custom Negative Prompts`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3158 | 0.6497 | 0.9655 |
| `decoritems_woodenpot` | 0.3551 | 0.7211 | 1.0762 |
| `furniture_sofa2` | 0.3021 | 0.8078 | 1.1099 |
| `instrument_music2` | 0.3509 | 0.7088 | 1.0597 |
| `luggage_backpack1` | 0.3236 | 0.7861 | 1.1097 |
| `person_3` | 0.3007 | 0.5667 | 0.8674 |
| `pet_cat5` | 0.3260 | 0.7812 | 1.1072 |
| `scene_waterfall` | 0.3388 | 0.7834 | 1.1222 |
| `transport_tank` | 0.3019 | 0.6537 | 0.9556 |
| `wearable_jacket1` | 0.3088 | 0.7752 | 1.0840 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3224** | **0.7234** | **1.0458** |

---

## 💡 3. 심층 결과 분석 및 고찰

수학적 2차 수치 적분(Heun)을 통해 고주파 외곽선과 텍스처를 정밀 복원하여 평균 CLIP-I 0.7234, furniture_sofa2에서 0.8078을 기록하며 최고 성능 달성.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/07_heun_custom_neg/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
