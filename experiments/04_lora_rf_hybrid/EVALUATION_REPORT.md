# 📊 Subject-driven LoRA + RF-Inversion Hybrid Evaluation Report (Iter 4)

- **실행 일시**: 2026-08-19 07:38:00
- **소요 시간**: 607.5초 (10.1분)
- **방법론**: `LoRA Fine-Tuning + Controlled ODE Inversion Hybrid`
- **하이퍼파라미터**: Steps=28, CFG=7.0, tau=0.7, eta=0.8, Token='sks'

## 1. 정량 평가 요약 (CLIP Scores)

| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | **0.2744** | **0.6749** | 0.9493 |
| `decoritems_woodenpot` | **0.3286** | **0.7854** | 1.1140 |
| `furniture_sofa2` | **0.2769** | **0.9011** | 1.1780 |
| `instrument_music2` | **0.3267** | **0.8128** | 1.1395 |
| `luggage_backpack1` | **0.3144** | **0.8554** | 1.1698 |
| `person_3` | **0.3038** | **0.5423** | 0.8461 |
| `pet_cat5` | **0.3126** | **0.8275** | 1.1401 |
| `scene_waterfall` | **0.3357** | **0.7871** | 1.1228 |
| `transport_tank` | **0.2986** | **0.6368** | 0.9354 |
| `wearable_jacket1` | **0.3105** | **0.8109** | 1.1214 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVG)** | **0.3082** | **0.7634** | **1.0716** |

