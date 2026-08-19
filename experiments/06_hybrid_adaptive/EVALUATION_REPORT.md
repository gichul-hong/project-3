# 📊 Subject-driven LoRA + RF-Inversion Hybrid Evaluation Report

- **실행 일시**: 2026-08-19 13:34:38
- **소요 시간**: 68.3초 (1.1분)
- **방법론**: `LoRA Fine-Tuning + Controlled ODE Inversion Hybrid (adaptive eta, avg ref)`
- **하이퍼파라미터**: Steps=28 (euler), CFG=7.0, tau=0.7, eta=0.8, Token='sks'

## 1. 정량 평가 요약 (CLIP Scores)

| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |
| :--- | :---: | :---: | :---: |
| `wearable_jacket1` | **0.3077** | **0.7694** | 1.0771 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVG)** | **0.3241** | **0.7192** | **1.0433** |

