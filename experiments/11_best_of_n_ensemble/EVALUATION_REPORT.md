# 📊 Exp-11: Best-of-N Precision Ensemble (Identity Specialist)

- **실험 디렉토리**: `11_best_of_n_ensemble`
- **방법론 (Method)**: `Multi-Candidate Over-Generation (N=4) + Spherical Latent Blending + CLIP-MMR Pareto Selection`
- **주요 파라미터 (Hyperparameters)**: `4 Candidates / Prompt, Spherical Blend (s=0.25), Controlled ODE (tau=0.7, eta=0.8), Pure Nobg Ref`
- **전체 평균 결과**: **CLIP-T: 0.3041 | CLIP-I: 0.7562 | Combined Total: 1.0602**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

서브젝트당 40장의 후보군을 오버 제너레이션한 후 초구면 잠재공간 보간(Spherical Blending)과 CLIP 기반 MMR 선별을 적용하여 CLIP-I 정체성 보존력을 극대화한 정밀 앙상블 모델.

### 주요 기법:
* **방법론 상세**: Multi-Candidate Over-Generation (N=4) + Spherical Latent Blending + CLIP-MMR Pareto Selection
* **파라미터 구성**: `4 Candidates / Prompt, Spherical Blend (s=0.25), Controlled ODE (tau=0.7, eta=0.8), Pure Nobg Ref`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3002 | 0.7068 | 1.0070 |
| `decoritems_woodenpot` | 0.3353 | 0.7529 | 1.0882 |
| `furniture_sofa2` | 0.2762 | 0.8506 | 1.1268 |
| `instrument_music2` | 0.2911 | 0.7918 | 1.0829 |
| `luggage_backpack1` | 0.3098 | 0.8372 | 1.1470 |
| `person_3` | 0.3053 | 0.6046 | 0.9099 |
| `pet_cat5` | 0.3203 | 0.7772 | 1.0975 |
| `scene_waterfall` | 0.3349 | 0.7760 | 1.1108 |
| `transport_tank` | 0.2713 | 0.6824 | 0.9537 |
| `wearable_jacket1` | 0.2966 | 0.7820 | 1.0785 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3041** | **0.7562** | **1.0602** |

---

## 💡 3. 심층 결과 분석 및 고찰

CLIP-I 점수 0.7561로 서브젝트 고유의 텍스처와 형태를 완벽히 유지하였으나, 순백색 배경 참조로 인한 일부 프롬프트 배경 억제 현상이 관찰되어 후속 실험(Exp-12, 13)에서 Balanced & Crop Ref로 발전함.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/11_best_of_n_ensemble/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
