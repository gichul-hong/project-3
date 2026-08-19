# 📊 Exp-03: Augmented SD3.5 LoRA Fine-Tuning (Rank 16, Steps 200)

- **실험 디렉토리**: `03_lora_augmented`
- **방법론 (Method)**: `SD3.5 LoRA Fine-Tuning on Background-Removed Augmented Dataset`
- **주요 파라미터 (Hyperparameters)**: `Rank=16, Alpha=32, LR=1e-4, Steps=200, Token='sks', Scheduler=Euler`
- **전체 평균 결과**: **CLIP-T: 0.3332 | CLIP-I: 0.6645 | Combined Total: 0.9977**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

Rembg 기반 배경 분리 증강 데이터셋을 사용하여 SD3Transformer2DModel의 Attention 레이어에 LoRA(Rank 16)를 파인튜닝.

### 주요 기법:
* **방법론 상세**: SD3.5 LoRA Fine-Tuning on Background-Removed Augmented Dataset
* **파라미터 구성**: `Rank=16, Alpha=32, LR=1e-4, Steps=200, Token='sks', Scheduler=Euler`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3224 | 0.4622 | 0.7846 |
| `decoritems_woodenpot` | 0.3563 | 0.6340 | 0.9903 |
| `furniture_sofa2` | 0.3248 | 0.7678 | 1.0926 |
| `instrument_music2` | 0.3507 | 0.7244 | 1.0751 |
| `luggage_backpack1` | 0.3389 | 0.7322 | 1.0711 |
| `person_3` | 0.3115 | 0.5163 | 0.8278 |
| `pet_cat5` | 0.3287 | 0.7944 | 1.1231 |
| `scene_waterfall` | 0.3438 | 0.7514 | 1.0952 |
| `transport_tank` | 0.3341 | 0.5599 | 0.8940 |
| `wearable_jacket1` | 0.3208 | 0.7021 | 1.0229 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3332** | **0.6645** | **0.9977** |

---

## 💡 3. 심층 결과 분석 및 고찰

배경 제거 증강으로 텍스트 프롬프트 준수력(CLIP-T 0.3332)이 크게 향상되었으나, 낮은 Rank(16)와 적은 Steps(200)로 인해 피사체 디테일(CLIP-I 0.6645)이 다소 감소.

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/03_lora_augmented/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
