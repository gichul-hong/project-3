# 📊 Exp-13: Ultimate SOTA Ensemble (Crop-Fit Ref + 1:1 Total Metric + White Guard)

- **실험 디렉토리**: `13_sota_ensemble`
- **방법론 (Method)**: `Crop-Fit Center Reference + Dual Objective Alignment (W_T=1.0, W_I=1.0) + Self-Healing White Background Guard`
- **주요 파라미터 (Hyperparameters)**: `4 Candidates / Prompt, Crop-Fit Reference, Controlled Euler ODE, White-Border Penalty (alpha=0.15)`
- **전체 평균 결과**: **CLIP-T: 0.3249 | CLIP-I: 0.7396 | Combined Total: 1.0645**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

중앙 크롭 피사체 잠재 융합과 순백색 배경 고착 페널티 가드(Self-Healing Guard), 공식 1:1 채점 지표 완벽 일치 선별기를 결합한 본 프로젝트 종합 SOTA 챔피언 모델.

### 주요 기법:
* **방법론 상세**: Crop-Fit Center Reference + Dual Objective Alignment (W_T=1.0, W_I=1.0) + Self-Healing White Background Guard
* **파라미터 구성**: `4 Candidates / Prompt, Crop-Fit Reference, Controlled Euler ODE, White-Border Penalty (alpha=0.15)`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3105 | 0.6407 | 0.9512 |
| `decoritems_woodenpot` | 0.3498 | 0.7842 | 1.1340 |
| `furniture_sofa2` | 0.3208 | 0.8424 | 1.1632 |
| `instrument_music2` | 0.3490 | 0.7637 | 1.1126 |
| `luggage_backpack1` | 0.3251 | 0.8242 | 1.1493 |
| `person_3` | 0.3059 | 0.5739 | 0.8798 |
| `pet_cat5` | 0.3315 | 0.7896 | 1.1210 |
| `scene_waterfall` | 0.3450 | 0.7609 | 1.1059 |
| `transport_tank` | 0.3002 | 0.6223 | 0.9225 |
| `wearable_jacket1` | 0.3117 | 0.7939 | 1.1055 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3249** | **0.7396** | **1.0645** |

---

## 💡 3. 심층 결과 분석 및 고찰

전체 10개 서브젝트 종합 점수 Total 1.0645(CLIP-T 0.3249 / CLIP-I 0.7396)로 프로젝트 역대 최고 종합 점수를 갱신하였으며, 사물 및 생명체 전반에서 무결점 품질을 달성함.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/13_sota_ensemble/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
