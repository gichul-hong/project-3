# 📊 Exp-01: RF-Inversion Baseline (Controlled ODE Single-Ref)

- **실험 디렉토리**: `01_rf_inversion_baseline`
- **방법론 (Method)**: `SD3.5 Base + Controlled ODE Inversion (Single Raw Reference)`
- **주요 파라미터 (Hyperparameters)**: `Steps=28, CFG=7.0, tau=0.7, eta=0.9, Scheduler=Euler`
- **전체 평균 결과**: **CLIP-T: 0.2950 | CLIP-I: 0.7831 | Combined Total: 1.0781**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

사전학습된 SD3.5-Medium 베이스 모델에 대해 단일 원본 레퍼런스 이미지를 Inversion하여 잠재 궤적을 제어하는 베이스라인.

### 주요 기법:
* **방법론 상세**: SD3.5 Base + Controlled ODE Inversion (Single Raw Reference)
* **파라미터 구성**: `Steps=28, CFG=7.0, tau=0.7, eta=0.9, Scheduler=Euler`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.2755 | 0.6950 | 0.9705 |
| `decoritems_woodenpot` | 0.3151 | 0.7575 | 1.0726 |
| `furniture_sofa2` | 0.2629 | 0.9221 | 1.1850 |
| `instrument_music2` | 0.2912 | 0.8181 | 1.1093 |
| `luggage_backpack1` | 0.3141 | 0.8628 | 1.1769 |
| `person_3` | 0.2813 | 0.6335 | 0.9148 |
| `pet_cat5` | 0.2928 | 0.8578 | 1.1506 |
| `scene_waterfall` | 0.3102 | 0.8171 | 1.1273 |
| `transport_tank` | 0.2999 | 0.6493 | 0.9492 |
| `wearable_jacket1` | 0.3069 | 0.8179 | 1.1248 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.2950** | **0.7831** | **1.0781** |

---

## 💡 3. 심층 결과 분석 및 고찰

원본 이미지의 픽셀/Latent 궤적을 직접 보정하므로 높은 피사체 보존력(CLIP-I 0.7831)을 보이나, 새로운 배경/스타일 생성 시 텍스트 프롬프트 준수력(CLIP-T 0.2950)에 한계가 있음.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/01_rf_inversion_baseline/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
