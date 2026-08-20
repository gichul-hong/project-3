# 📊 Subject-driven LoRA + RF-Inversion Hybrid Evaluation Report

- **실행 일시**: 2026-08-20 02:59:43
- **소요 시간**: 142.6초 (2.4분)
- **방법론**: `LoRA Fine-Tuning + Controlled ODE Inversion Hybrid (adaptive eta, avg ref)`
- **하이퍼파라미터**: Steps=50 (heun), CFG=7.0, tau=0.7, eta=0.8, Token='sks'

## 1. 정량 평가 요약 (CLIP Scores)

| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |
| :--- | :---: | :---: | :---: |
| `luggage_backpack1` | **0.3286** | **0.7670** | 1.0956 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVG)** | **0.3359** | **0.6932** | **1.0291** |

