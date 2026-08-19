# 📊 Exp-04: LoRA + Controlled ODE Hybrid (Rank 16)

- **실험 디렉토리**: `04_lora_rf_hybrid`
- **방법론 (Method)**: `SD3.5 LoRA (Rank 16) + Controlled ODE Single-Ref Inversion 결합`
- **주요 파라미터 (Hyperparameters)**: `LoRA Rank 16 + Steps=28, CFG=7.0, tau=0.7, eta=0.8, Scheduler=Euler`
- **전체 평균 결과**: **CLIP-T: 0.3082 | CLIP-I: 0.7634 | Combined Total: 1.0716**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

LoRA의 프롬프트 적응력과 Controlled ODE의 정체성 제어 궤적을 결합한 하이브리드 파이프라인.

### 주요 기법:
* **방법론 상세**: SD3.5 LoRA (Rank 16) + Controlled ODE Single-Ref Inversion 결합
* **파라미터 구성**: `LoRA Rank 16 + Steps=28, CFG=7.0, tau=0.7, eta=0.8, Scheduler=Euler`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.2744 | 0.6749 | 0.9493 |
| `decoritems_woodenpot` | 0.3286 | 0.7854 | 1.1140 |
| `furniture_sofa2` | 0.2769 | 0.9011 | 1.1780 |
| `instrument_music2` | 0.3267 | 0.8128 | 1.1395 |
| `luggage_backpack1` | 0.3144 | 0.8554 | 1.1698 |
| `person_3` | 0.3038 | 0.5423 | 0.8461 |
| `pet_cat5` | 0.3126 | 0.8275 | 1.1401 |
| `scene_waterfall` | 0.3357 | 0.7871 | 1.1228 |
| `transport_tank` | 0.2986 | 0.6368 | 0.9354 |
| `wearable_jacket1` | 0.3105 | 0.8109 | 1.1214 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3082** | **0.7634** | **1.0716** |

---

## 💡 3. 심층 결과 분석 및 고찰

LoRA 단독 대비 피사체 보존력(CLIP-I 0.7634)이 대폭 복원되며 CLIP-T(0.3082)와 균형을 이룸.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/04_lora_rf_hybrid/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
