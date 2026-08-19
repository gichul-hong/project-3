# 📊 Exp-06: Hybrid Multi-Reference Inversion Averaging + Adaptive eta

- **실험 디렉토리**: `06_hybrid_adaptive`
- **방법론 (Method)**: `LoRA HQ + Controlled ODE (Multi-Ref Inversion Avg + Cosine Adaptive eta)`
- **주요 파라미터 (Hyperparameters)**: `LoRA Rank 64 + Multi-Ref Avg + Cosine eta (0.8->0.0) + tau=0.7, Steps=28, CFG=7.0`
- **전체 평균 결과**: **CLIP-T: 0.3241 | CLIP-I: 0.7192 | Combined Total: 1.0433**

---

## 📝 1. 실험 개요 및 핵심 메커니즘

원본+배경제거 이미지 N장의 Inversion Latent를 앙상블 평균(Multi-reference Averaging)하고, 생성 후반부 프롬프트 자유도를 보장하는 Adaptive eta 스케줄링 적용.

### 주요 기법:
* **방법론 상세**: LoRA HQ + Controlled ODE (Multi-Ref Inversion Avg + Cosine Adaptive eta)
* **파라미터 구성**: `LoRA Rank 64 + Multi-Ref Avg + Cosine eta (0.8->0.0) + tau=0.7, Steps=28, CFG=7.0`

---

## 📈 2. 10개 서브젝트별 정량 평가 결과 (CLIP Scores)

| 서브젝트 (Concept) | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined Score (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | 0.3177 | 0.6505 | 0.9682 |
| `decoritems_woodenpot` | 0.3557 | 0.7186 | 1.0743 |
| `furniture_sofa2` | 0.3175 | 0.7997 | 1.1172 |
| `instrument_music2` | 0.3544 | 0.7099 | 1.0643 |
| `luggage_backpack1` | 0.3212 | 0.7755 | 1.0967 |
| `person_3` | 0.3068 | 0.5653 | 0.8721 |
| `pet_cat5` | 0.3283 | 0.7888 | 1.1171 |
| `scene_waterfall` | 0.3349 | 0.7789 | 1.1138 |
| `transport_tank` | 0.2966 | 0.6350 | 0.9316 |
| `wearable_jacket1` | 0.3077 | 0.7694 | 1.0771 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVERAGE)** | **0.3241** | **0.7192** | **1.0433** |

---

## 💡 3. 심층 결과 분석 및 고찰

단일 레퍼런스의 배경 바이어스를 완벽히 제거하고 피사체 공통 불변 특징만 주입하여, Exp-05 대비 CLIP-I가 0.6731 ➔ 0.7192 (+0.046p, +6.8%)로 비약적 상승!

---

## 📁 4. 생성 산출물 및 재실행 안내

* **생성 이미지 디렉토리**: `./experiments/06_hybrid_adaptive/[concept_name]/` (서브젝트당 10장, 총 100장)
* **웹 대시보드 뷰어**: [experiment_viewer.html](file:///content/project-3/experiment_viewer.html)
