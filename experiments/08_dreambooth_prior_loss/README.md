# 🧪 Experiment: 08_dreambooth_prior_loss

## 1. 실험 개요 및 방법론
- **방법론**: `SD3.5 LoRA + Controlled ODE RF-Inversion Hybrid`
- **스케줄러**: `heun` (50 steps)
- **Controlled ODE 설정**: tau=0.7, eta=0.85, gamma=0.5, eta_schedule=adaptive
- **Reference 모드**: `avg` (Multi-reference Latent Ensemble)
- **T5-XXL 텍스트 인코더**: 활성화
- **Custom Negative Prompt**: 적용

## 2. 재실행(Reproduction) 명령어
```bash
python generate_hybrid.py \
    --concept all \
    --checkpoints_dir ./checkpoints/exp08_dreambooth_lora \
    --output ./experiments/08_dreambooth_prior_loss \
    --ref_mode avg \
    --eta_schedule adaptive \
    --scheduler heun \
    --tau 0.7 \
    --eta 0.85 \
    --steps 50 \
    --enable_t5 \
    --custom_neg
```

## 3. 평가 점수 요약
- **Text-to-Image (CLIP-T)**: **0.3327**
- **Image-to-Image (CLIP-I)**: **0.7014**
- **Total Combined (T+I)**: **1.0341**

> 📌 상세 10개 서브젝트별 22개 점수표: [`EVALUATION_REPORT.md`](file:///content/project-3/experiments/08_dreambooth_prior_loss/EVALUATION_REPORT.md)
