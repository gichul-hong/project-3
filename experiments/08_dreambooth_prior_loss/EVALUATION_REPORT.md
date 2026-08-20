# 📊 Subject-driven LoRA + RF-Inversion Hybrid Evaluation Report

- **실행 일시**: 2026-08-20 02:38:23
- **소요 시간**: 148.7초 (2.5분)
- **방법론**: `LoRA Fine-Tuning + Controlled ODE Inversion Hybrid (adaptive eta, avg ref)`
- **하이퍼파라미터**: Steps=50 (heun), CFG=7.0, tau=0.7, eta=0.85, Token='sks'

## 1. 정량 평가 요약 (CLIP Scores)

| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |
| :--- | :---: | :---: | :---: |
| `pet_cat5` | **0.3240** | **0.7842** | 1.1082 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVG)** | **0.3301** | **0.6891** | **1.0192** |

