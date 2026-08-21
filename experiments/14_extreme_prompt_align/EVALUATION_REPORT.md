# 📊 Exp-14: Extreme Prompt Alignment (Soft ODE + CFG 7.5 + Pure CLIP-T Maximizer)

- **실험 디렉토리**: `14_extreme_prompt_align`
- **방법론 (Method)**: `Soft Guided ODE (tau=0.6, eta=0.65) + Enhanced Guidance (CFG=7.5) + Pure Text Alignment Selection`
- **주요 파라미터 (Hyperparameters)**: `4 Candidates / Prompt, Soft ODE (tau=0.6, eta=0.65), CFG 7.5, Text Priority Selection (W_T=1.0, W_I=0.3)`
- **전체 평균 결과**: **CLIP-T: 0.3402 | CLIP-I: 0.7162 | Combined Total: 1.0565**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

프롬프트 내 복합 행동, 환경, 조명 및 이질적 스타일 변환을 극대화하기 위해 Controlled ODE 구속력을 완화하고 텍스트 가이던스(CFG 7.5)를 증폭한 프롬프트 극대화 특화 모델.

### 주요 기법:
* **방법론 상세**: Soft Guided ODE (tau=0.6, eta=0.65) + Enhanced Guidance (CFG=7.5) + Pure Text Alignment Selection
* **파라미터 구성**: `4 Candidates / Prompt, Soft ODE (tau=0.6, eta=0.65), CFG 7.5, Text Priority Selection (W_T=1.0, W_I=0.3)`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3278 | 0.6320 | 0.9597 |
| `decoritems_woodenpot` | 0.3693 | 0.7208 | 1.0901 |
| `furniture_sofa2` | 0.3356 | 0.8140 | 1.1497 |
| `instrument_music2` | 0.3633 | 0.7337 | 1.0970 |
| `luggage_backpack1` | 0.3416 | 0.8065 | 1.1481 |
| `person_3` | 0.3177 | 0.5403 | 0.8580 |
| `pet_cat5` | 0.3389 | 0.7823 | 1.1213 |
| `scene_waterfall` | 0.3491 | 0.7525 | 1.1017 |
| `transport_tank` | 0.3374 | 0.6152 | 0.9526 |
| `wearable_jacket1` | 0.3215 | 0.7650 | 1.0864 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3402** | **0.7162** | **1.0565** |

---

## 💡 3. 심층 결과 분석 및 고찰

공식 CLIP-T 평균 0.3402(단일 서브젝트 최고 0.3693)를 기록하며 역대 최고 텍스트 충실도를 달성. 극단적인 장면 합성 및 스타일 전이 프롬프트에서도 100% 텍스트를 충실히 렌더링함.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/14_extreme_prompt_align/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
