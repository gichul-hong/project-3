# 📊 Exp-09: SOTA Final Ensemble - Subject-Aware Dynamic tau/eta Routing ODE

- **실험 디렉토리**: `09_subject_adaptive_routing`
- **방법론 (Method)**: `DreamBooth-LoRA + Subject-Aware Dynamic Guidance Routing (Rigid vs Flexible Routing)`
- **주요 파라미터 (Hyperparameters)**: `Rigid (tau=0.75, eta=0.90) / Flexible (tau=0.60, eta=0.70) + Heun 50 Steps`
- **전체 평균 결과**: **CLIP-T: 0.3268 | CLIP-I: 0.6908 | Combined Total: 1.0176**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

서브젝트의 물리적 특성(사물/가구 Rigid vs 인물/동물 Flexible)을 인지하여 Controlled ODE 가이던스 강도와 임계점을 동적 분기 라우팅하는 최종 완성형 SOTA 모델.

### 주요 기법:
* **방법론 상세**: DreamBooth-LoRA + Subject-Aware Dynamic Guidance Routing (Rigid vs Flexible Routing)
* **파라미터 구성**: `Rigid (tau=0.75, eta=0.90) / Flexible (tau=0.60, eta=0.70) + Heun 50 Steps`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3396 | 0.5460 | 0.8856 |
| `decoritems_woodenpot` | 0.3616 | 0.6646 | 1.0262 |
| `furniture_sofa2` | 0.2992 | 0.7954 | 1.0946 |
| `instrument_music2` | 0.3503 | 0.6929 | 1.0432 |
| `luggage_backpack1` | 0.3286 | 0.7670 | 1.0956 |
| `person_3` | 0.3001 | 0.5530 | 0.8531 |
| `pet_cat5` | 0.3259 | 0.7709 | 1.0968 |
| `scene_waterfall` | 0.3430 | 0.7869 | 1.1299 |
| `transport_tank` | 0.2979 | 0.5774 | 0.8753 |
| `wearable_jacket1` | 0.3220 | 0.7539 | 1.0759 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3268** | **0.6908** | **1.0176** |

---

## 💡 3. 심층 결과 분석 및 고찰

사물에서는 강력한 외형 보존을 유지하고 인물/동물에서는 자연스러운 포즈/배경 변형 자유도를 부여하여 전체 10개 서브젝트의 CLIP-T/CLIP-I 종합 점수 최고점을 갱신.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/09_subject_adaptive_routing/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
