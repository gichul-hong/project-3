# 📊 Exp-05: High-Quality LoRA Fine-Tuning (T5-XXL + Rank 64, 1000 Steps)

- **실험 디렉토리**: `05_lora_hq`
- **방법론 (Method)**: `SD3.5 High-Rank LoRA with T5-XXL Text Encoder Active`
- **주요 파라미터 (Hyperparameters)**: `Rank=64, Alpha=64, LR=5e-5, Steps=1000, T5-XXL=Active, Steps_gen=28, CFG=7.0`
- **전체 평균 결과**: **CLIP-T: 0.3239 | CLIP-I: 0.6731 | Combined Total: 0.9970**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

A100 40GB 환경을 활용하여 T5-XXL(4.7B) 텍스트 인코더를 전면 활성화하고, LoRA Rank를 64로 4배 확장, 1,000 Steps 충분 수렴 학습.

### 주요 기법:
* **방법론 상세**: SD3.5 High-Rank LoRA with T5-XXL Text Encoder Active
* **파라미터 구성**: `Rank=64, Alpha=64, LR=5e-5, Steps=1000, T5-XXL=Active, Steps_gen=28, CFG=7.0`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3109 | 0.5808 | 0.8917 |
| `decoritems_woodenpot` | 0.3637 | 0.6616 | 1.0253 |
| `furniture_sofa2` | 0.3044 | 0.7787 | 1.0831 |
| `instrument_music2` | 0.3522 | 0.6925 | 1.0447 |
| `luggage_backpack1` | 0.3288 | 0.7320 | 1.0608 |
| `person_3` | 0.3055 | 0.5151 | 0.8206 |
| `pet_cat5` | 0.3227 | 0.7361 | 1.0588 |
| `scene_waterfall` | 0.3332 | 0.7472 | 1.0804 |
| `transport_tank` | 0.3016 | 0.5723 | 0.8739 |
| `wearable_jacket1` | 0.3164 | 0.7142 | 1.0306 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3239** | **0.6731** | **0.9970** |

---

## 💡 3. 심층 결과 분석 및 고찰

T5-XXL 텍스트 인코더와 Rank 64 확장으로 세부 질감 및 복합 속성 이해도가 대폭 상승하여 Exp-03 대비 전반적인 생성 품질과 정체성(CLIP-I 0.6731)이 개선됨.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/05_lora_hq/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
