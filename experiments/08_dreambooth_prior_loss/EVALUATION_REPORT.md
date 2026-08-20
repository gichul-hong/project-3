# 📊 Subject-driven LoRA + RF-Inversion Hybrid Evaluation Report

- **실행 일시**: 2026-08-20 02:22:25
- **소요 시간**: 151.0초 (2.5분)
- **방법론**: `LoRA Fine-Tuning + Controlled ODE Inversion Hybrid (adaptive eta, avg ref)`
- **하이퍼파라미터**: Steps=50 (heun), CFG=7.0, tau=0.7, eta=0.85, Token='sks'

## 1. 정량 평가 요약 (CLIP Scores)

| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | **0.3396** | **0.5460** | 0.8856 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVG)** | **0.3396** | **0.5460** | **0.8856** |

