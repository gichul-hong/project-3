# 🧪 Experiment: 05_lora_hq

## 1. 실험 개요 및 방법론
- **방법론**: `SD3.5 High-Quality DreamBooth LoRA (T5-XXL + Rank 64)`
- **데이터셋**: `./augmentation` (5종 증강 + nobg 가중 전처리)
- **학습 하이퍼파라미터**: Rank=64, Alpha=64, Steps=1000, LR=5e-5, T5-XXL Active
- **생성 하이퍼파라미터**: Steps=28, CFG=7.0, Custom Negative Prompt=True

## 2. 재실행(Reproduction) 명령어
```bash
# 1) LoRA 파인튜닝 학습
python train_lora_sd3.py --concept all --exp_name exp05_lora_hq --rank 64 --alpha 64 --steps 1000 --lr 5e-5 --enable_t5

# 2) 100장 생성 및 CLIP-B/32 자동 채점
python generate_lora.py --concept all --exp_name exp05_lora_hq --output ./experiments/05_lora_hq --steps 28 --enable_t5 --custom_neg
```

## 3. 평가 점수 요약
- **Text-to-Image (CLIP-T)**: **0.3239**
- **Image-to-Image (CLIP-I)**: **0.6731**
- **Total Combined (T+I)**: **0.9970**

> 📌 상세 10개 서브젝트별 22개 점수표: [`EVALUATION_REPORT.md`](file:///content/project-3/experiments/05_lora_hq/EVALUATION_REPORT.md)
