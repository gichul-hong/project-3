# 📊 Subject-driven SD3.5 LoRA Fine-Tuning Evaluation Report

- **실행 일시**: 2026-08-19 07:04:05
- **소요 시간**: 562.0초 (9.4분)
- **방법론**: `SD3.5 LoRA Fine-Tuning (Augmented Dataset)`
- **하이퍼파라미터**: Steps=28, CFG=7.0, Token='sks', Seed=42

## 1. 정량 평가 요약 (CLIP Scores)

| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | **0.3224** | **0.4622** | 0.7846 |
| `decoritems_woodenpot` | **0.3563** | **0.6340** | 0.9903 |
| `furniture_sofa2` | **0.3248** | **0.7678** | 1.0926 |
| `instrument_music2` | **0.3507** | **0.7244** | 1.0751 |
| `luggage_backpack1` | **0.3389** | **0.7322** | 1.0711 |
| `person_3` | **0.3115** | **0.5163** | 0.8278 |
| `pet_cat5` | **0.3287** | **0.7944** | 1.1231 |
| `scene_waterfall` | **0.3438** | **0.7514** | 1.0952 |
| `transport_tank` | **0.3341** | **0.5599** | 0.8940 |
| `wearable_jacket1` | **0.3208** | **0.7021** | 1.0229 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVG)** | **0.3332** | **0.6645** | **0.9977** |

