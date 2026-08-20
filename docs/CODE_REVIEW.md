# 코드 리뷰 — 실험 파이프라인 문제점 및 개선안 (2026-08-20)

> 대상: `generate_hybrid.py`, `run_exp11_ensemble.py`, `train_lora_sd3.py`, `train_dreambooth_sd3.py`, `evaluate_extended.py`
> 기준 문서: `docs/EXPERIMENT_HISTORY.md` (Exp-01~09), `experiments/11_best_of_n_ensemble/` (중단됨)

---

## 0. 현재 상태

- **exp11 중단**: `actionfigure_2`만 완료(10장+selection.json). `decoritems_woodenpot` 후보 36/40에서 중단. 나머지 8개 subject 미생성.
- **공식 점수 1위는 여전히 Exp-01** (T 0.2950 / I 0.7831 / Total 1.0781).
- 로컬 파일과 Colab 실행본이 다를 수 있음 (C1 버그가 있는데 exp11 산출물 존재 → Colab 쪽 코드가 수정본일 가능성).

## 1. 치명 (Critical)

### C1. `torch.randn_like(x, generator=...)` — TypeError 크래시
- `generate_hybrid.py:467`, `run_exp11_ensemble.py:146`
- torch는 `randn_like`에 `generator` 인자를 지원하지 않음.
- 수정: `torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)`

### C2. `ControlledHeunODE`는 Heun이 아님 (가짜 2차)
- `generate_hybrid.py:176-235`
- `FlowMatchHeunDiscreteScheduler` 상속 후 `step()`을 1차 Euler로 오버라이드.
  부모의 `set_timesteps()`는 sigma를 interleave(2N-1개)로 만들기 때문에
  절반의 스텝이 `sigma_next == sigma`(dt=0) no-op에 풀 모델 평가를 낭비.
- 결과: "Heun 50 steps" = Euler 50 유효 스텝 + 연산 2배.
- Exp-07/08/09/11 보고서의 "2nd-Order Heun Predictor-Corrector" 주장 수정 필요.
- 수정 옵션: (a) FlowMatchEulerDiscreteScheduler 기반으로 바꾸고 스텝 절반 (동일 품질, 2배 빠름),
  (b) 진짜 Heun predictor-corrector 구현 (부모 step 로직에 controller 주입).

### C3. evaluate_extended.py 페어링이 파일명 정렬 순서 의존
- `evaluate_extended.py:251, 282-286, 305-314`
- CLIP-T와 taxonomy가 "정렬된 파일명 == 프롬프트 인덱스, 1:1" 가정.
  N-per-prompt나 `10.png` 정렬 문제 시 점수 오염 (조용히 진행).
  `len(prompts) < len(images)`이면 shape mismatch 크래시.
- EXPERIMENT_HISTORY.md 2-2절(확장 평가표) 수치는 재검증 권장.

### C4. exp11 마지막 공식 평가 호출 무조건 실패
- `run_exp11_ensemble.py:277`
- `evaluation.py`는 `--concept` required인데 누락. `python3` + `os.system`은 Windows 비호환.
- 수정: subject 루프를 돌며 `--concept {s} --images {out}/{s}`로 호출, `sys.executable` 사용.

## 2. 중요 (Important)

1. **PEFT 키 접두사** — `train_lora_sd3.py:352-356`, `train_dreambooth_sd3.py:391-395`
   `get_peft_model()` 상태에서 `get_peft_model_state_dict()` → `base_model.model.` 접두사.
   `load_lora_weights()` 로드 실패 가능. try/except가 오류를 삼킴.
   → 저장된 checkpoint 1개를 실제 로드 테스트로 검증할 것.
2. **logit-normal 샘플링에 shift 미적용** — `train_lora_sd3.py:290-298`, `train_dreambooth_sd3.py:297-312`
   SD3.5-medium 추론 스케줄(shift≈3)과 학습 노이즈 분포 불일치.
   dreambooth 쪽은 CLI에서 logit-normal 선택 자체가 불가.
3. **DreamBooth λ 불일치** — docstring/함수 기본 1.0 vs CLI 기본 `--prior_weight 0.3`.
   Exp-08 실제 학습 λ 확인 후 발표자료에 명시.
4. **Inversion timestep off-by-one** — `generate_hybrid.py:248-265`
   스텝 i에서 구간 종점 시각으로 모델 조건화. 체계적 inversion 오차.
5. **`blend_anchor` 분산 축소** — `run_exp11_ensemble.py:72-77`
   `(1-s)a + s·n`은 std를 √((1-s)²+s²)<1로 축소 → 흐릿한 후보.
   수정: `√(1-s²)·a + s·n` (구면 보간). `ci=0`은 strength 0이라 후보 1개가 항상 동일.
6. **죽은 per-prompt seed** — `generate_hybrid.py:510`
   `latents=` 전달 시 generator 무시 → 10개 프롬프트가 동일 초기 latent.
   (pet_cat5 "총 든 고양이"류 아티팩트가 전 프롬프트에 전파되는 구조적 원인)
7. **tau 표기 반전** — 코드 tau=0.7 ≒ 논문 τ=0.3 (sigma=1-t 규약 차이). 발표 인용 시 주의.
8. **resume 불가** — `run_exp11_ensemble.py`에 SKIP_DONE 없음. selection.json 존재 시 스킵 추가.
9. **EXPERIMENT_HISTORY.md LaTeX 깨짐** — `$	au$`(탭), `rac`(\f 소실).
   리포트 생성 스크립트(update_*_reports.py)의 문자열 이스케이프 버그. raw string 사용.
10. **cosine eta 스케줄이 Heun 클래스에 없음** — `--eta_schedule cosine --scheduler heun` 시 무음 폴백.
11. **concept마다 전체 파이프라인(T5 포함) 재로딩** — `generate_hybrid.py:597-632`.
    exp11처럼 LoRA만 교체하는 패턴으로 통일하면 로딩 시간 1/10.

## 3. 사소 (Minor, 선별)

- `nobg` 모드가 `*_nobg.png`만 검색 (jpg 미지원) — augmentation과 확장자 일관성.
- VAE `posterior.sample()` 고정 seed 1회 캐싱 — 레퍼런스 latent에는 `mode()`가 표준.
- 512×512 강제 리사이즈 (SD3.5 네이티브 1024) — 왜곡 유입 가능. 의도라면 발표에 명시.
- `--custom_neg`/`--enable_t5`가 default=True + store_true → 끌 방법 없음 (실험 변인 통제 실패).
- eval_summary.json에 subject_routing 실제 값 대신 전역 tau/eta 기록 — 보고서 불일치.
- cosine LR 스케줄러가 grad accumulation 때문에 절반만 진행.
- `evaluate_extended.py`: DINOv2 torch.hub 로드(버전 미고정, 실패 시 0.0으로 조용히 보고),
  `open()` 인코딩 미지정(cp949 위험), `p.replace("sks","")`가 단어 내부까지 치환,
  `experiments/EXTENDED_COMPARISON.json` 하드코딩.

## 4. 우선순위 실행 계획

| 순위 | 작업 | 기대 효과 | 예상 시간 |
|---|---|---|---|
| 1 | C1+C4 수정, resume 추가 → exp11 전체 재실행 | Best-of-N은 공식 점수 직결 (선별로 T/I 동시 최적화) | 수정 30분 + A100 3h |
| 2 | blend_anchor 분산 보정 + ci=0 미세노이즈 | 후보 품질/다양성 즉시 개선 | 10분 |
| 3 | C2 처리: Euler 기반 전환(스텝 절반) 또는 발표 표현 수정 | 2배 속도 or 정확한 보고 | 20분 |
| 4 | PEFT 로드 테스트 (checkpoint 1개) | Exp-05/08 재현성 확인 | 10분 |
| 5 | EXPERIMENT_HISTORY.md LaTeX/Exp-11 갱신 | 발표자료 품질 | 20분 |
| 6 | C3 수정 or 확장표에 주석 | 확장 지표 신뢰성 | 시간 남으면 |

## 5. 방법론 참고 (수식 검증 결과)

- RF-Inversion 핵심 수식은 **정확**: inversion `(z-x_t)/(1-t)`, generation `(x_t-x0)/t`,
  controlled velocity `v+η(v_c-v)`, null-prompt+CFG=1 inversion 모두 논문과 일치.
- Flow Matching 학습 loss도 **정확**: `x_t=(1-σ)x0+σε`, `target=ε-x0`, `t=σ·1000`.
- 텍스트 임베딩/VAE latent 사전 캐싱은 모범적으로 구현됨.
