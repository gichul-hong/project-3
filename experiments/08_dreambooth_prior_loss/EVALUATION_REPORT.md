# 📊 Subject-driven LoRA + RF-Inversion Hybrid Evaluation Report

- **실행 일시**: 2026-08-20 02:27:40
- **소요 시간**: 146.0초 (2.4분)
- **방법론**: `LoRA Fine-Tuning + Controlled ODE Inversion Hybrid (adaptive eta, avg ref)`
- **하이퍼파라미터**: Steps=50 (heun), CFG=7.0, tau=0.7, eta=0.85, Token='sks'

## 1. 정량 평가 요약 (CLIP Scores)

| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |
| :--- | :---: | :---: | :---: |
| `furniture_sofa2` | **0.2986** | **0.7952** | 1.0938 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVG)** | **0.3344** | **0.6682** | **1.0026** |

