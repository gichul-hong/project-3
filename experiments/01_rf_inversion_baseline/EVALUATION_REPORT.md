# 📊 Subject-driven RF-Inversion Evaluation Report

- **실행 일시**: 2026-08-19 05:50:23
- **소요 시간**: 433.4초 (7.2분)
- **방법론**: `RF-Inversion` (Ref Mode: `first`)
- **데이터셋 경로**: `./dataset`
- **하이퍼파라미터**: Steps=28, CFG=7.0, tau=0.7, eta=0.9, seed=42

## 1. 정량 평가 요약 (CLIP Scores)

| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | **0.2755** | **0.6950** | 0.9705 |
| `decoritems_woodenpot` | **0.3151** | **0.7575** | 1.0726 |
| `furniture_sofa2` | **0.2629** | **0.9221** | 1.1850 |
| `instrument_music2` | **0.2912** | **0.8181** | 1.1093 |
| `luggage_backpack1` | **0.3141** | **0.8628** | 1.1769 |
| `person_3` | **0.2813** | **0.6335** | 0.9148 |
| `pet_cat5` | **0.2928** | **0.8578** | 1.1506 |
| `scene_waterfall` | **0.3102** | **0.8171** | 1.1273 |
| `transport_tank` | **0.2999** | **0.6493** | 0.9492 |
| `wearable_jacket1` | **0.3069** | **0.8179** | 1.1248 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVG)** | **0.2950** | **0.7831** | **1.0781** |

> 💡 **발표 자료 팁**: ProjectOverview 요구사항에 따라 서브젝트별 10개 값 + 전체 평균 1개 = 총 22개 수치로 정리되어 있습니다.
