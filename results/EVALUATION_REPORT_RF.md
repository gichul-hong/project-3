# 📊 Subject-driven RF-Inversion Evaluation Report

- **실행 일시**: 2026-08-19 05:42:34
- **소요 시간**: 50.1초 (0.8분)
- **방법론**: `RF-Inversion` (Ref Mode: `first`)
- **데이터셋 경로**: `./dataset`
- **하이퍼파라미터**: Steps=28, CFG=7.0, tau=0.7, eta=0.9, seed=42

## 1. 정량 평가 요약 (CLIP Scores)

| Concept | Text-to-Image (CLIP-T) | Image-to-Image (CLIP-I) | Combined (T+I) |
| :--- | :---: | :---: | :---: |
| `actionfigure_2` | **0.2755** | **0.6950** | 0.9705 |
| :--- | :---: | :---: | :---: |
| **전체 평균 (TOTAL AVG)** | **0.2755** | **0.6950** | **0.9705** |

> 💡 **발표 자료 팁**: ProjectOverview 요구사항에 따라 서브젝트별 10개 값 + 전체 평균 1개 = 총 22개 수치로 정리되어 있습니다.
