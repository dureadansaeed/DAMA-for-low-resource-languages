# DAMA on Whisper-small: Phase 1 reproduction

Members: Ritaj Suleman (24i-0112), Dur E Adan (24i-3037) | Course: Machine Learning Course Project, Sir Ahmed Raza

This repository reproduces, at a smaller scale, the experiments of the base paper
**"Depth-Aware Model Adaptation" (DAMA), Findings of ACL 2026**. The paper adapts Whisper large-v2 to
low-resource languages. We use **Whisper-small** (`openai/whisper-small`) instead, because it fits our single GPU.
We run the paper's 8 adaptation methods on its 18 languages (20 language runs, because Luganda and
Central Kurdish appear in both datasets), with two random seeds, and report our own numbers next to the paper's.

Methods: full fine-tuning (decoder only), LoRA, DoRA, LoRA-FA, LoRA-XS, AdaLoRA, VB-LoRA and DAMA. All adapters
go on the decoder only, as in the paper.

## Results in one table

Average WER (lower is better) over 20 language runs. `ALL` is seed 0 and `ALL_s1` is seed 1. `adapterM` is the number of
trained adapter parameters in millions (the token-embedding matrix is not counted, see "Notes").
Full tables are in `results/`.

```
method       CV  FLseen  FLunsn    ALL  ALL_s1  adapterM    GB   min
adalora    47.9    35.3    51.6   45.7    45.6      11.5   3.9   5.0
dama       47.7    35.3    50.9   45.4    45.6       3.5   3.2   3.8
dora       44.9    33.3    49.7   43.2    43.2      15.5   4.6   4.6
ft         47.9    36.5    51.8   46.0    45.7     153.6   4.5   4.2
lora       45.4    33.4    49.7   43.5    43.6      15.3   3.5   3.9
lorafa     48.6    36.4    51.9   46.4    46.6       7.7   3.3   3.8
loraxs     57.7    42.2    56.6   53.6    53.2       0.5   3.2   3.6
vblora     59.6    44.2    58.0   55.3    55.1       1.9   4.1   4.8
```

CV = Common Voice, FLseen = FLEURS languages Whisper already knows, FLunsn = FLEURS unseen languages,
GB = peak GPU memory, min = minutes per run.

## Environment

Tested on Ubuntu with Python 3.14, one NVIDIA RTX 4000 Ada (20 GB), driver 595.99.02, CUDA 13.2.
Main libraries: torch 2.15.0, transformers 5.22.2, peft 0.21.2, librosa 1.0.0, jiwer 4.0.0.
`requirements.txt` is the full `pip freeze` of the working environment. `ffmpeg` must be installed
(`sudo apt install ffmpeg`).

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data (not included)

1. **Common Voice subset (10 languages).** Download `CL-MASR.tar.gz` (5.9 GB) from the CL-MASR Zenodo record
   <https://zenodo.org/records/8065754> (md5 `810aa5d24ddacd3bf4c870b2c577f9c4`). Extract it so that
   `data/cv/CL-MASR/<lang>/` exists for rw, eo, kab, lg, mhr, ckb, ab, kmr, fy-NL and ia.
2. **FLEURS (10 languages).** `python download_fleurs.py` downloads `google/fleurs` for hi_in, cy_gb, be_by, fa_ir, sw_ke,
   lg_ug, ckb_iq, ast_es, ceb_ph and or_in into `data/fleurs/data/`.
3. `python check_data.py` prints clip counts and durations per language as a sanity check.
4. `python build_manifests.py` writes `manifests/*.jsonl` (60 files: 20 language runs x train/dev/test). Training data per
   language is capped to the paper's durations (10 hours or less); dev and test are capped at 1 hour each.

Dataset licences: Common Voice is CC0; FLEURS is CC BY 4.0.

## Reproducing the experiments

All commands run from the repository root with the virtual environment active. Every run writes one JSON file to
`runs2/` named `{dataset}_{language}_{method}{tag}.json`. Finished runs are skipped, so any command can be
restarted safely.

**Main run (seed 0, all methods, all 20 language runs; overnight on one GPU).** `go.sh` runs a short smoke test and then starts
the full run in the background (edit the `~/dama` path at the top of the script if the repository is elsewhere):

```
bash go.sh
# or directly:
python dama_repro.py --methods lora dama ft lorafa dora loraxs adalora vblora
```

**Second seed (seed 1):**

```
SEED=1 python dama_repro.py --methods lora dama dora ft --tag _s1
SEED=1 python dama_repro.py --methods adalora lorafa loraxs vblora --tag _s1
```

**Zero-shot baseline (the 5 FLEURS languages Whisper already knows):**

```
python dama_repro.py --methods zeroshot --langs fleurs:hi_in fleurs:cy_gb fleurs:be_by fleurs:fa_ir fleurs:sw_ke
```

**Learning-rate sweep (3 languages, 4 methods):**

```
for lr in 2e-4 5e-4 1e-3 2e-3; do
python dama_repro.py --methods lora dama loraxs vblora --langs cv:ia cv:kmr fleurs:or_in --lr $lr --tag _lr$lr
done
```

**DAMA capacity test (3 languages):**

```
RH=64 RL=16 python dama_repro.py --methods dama --langs cv:ia cv:kmr fleurs:or_in --lr 5e-4 --tag _cap64
RH=128 RL=32 python dama_repro.py --methods dama --langs cv:ia cv:kmr fleurs:or_in --lr 5e-4 --tag _cap128
```

**DAMA ablations.** `ABL=uniform` is plain LoRA at rank 16 (no SVD init, nothing frozen); `ABL=nosvd` uses random instead
of SVD init; `ABL=uniformrank` uses rank 16 everywhere but keeps SVD init and the frozen middle layers; `ABL=nofreeze`
trains the middle-layer A matrices. The first three ran on all 20 language runs, `nofreeze` on 3 languages only:

```
for ab in uniform nosvd uniformrank; do
ABL=$ab python dama_repro.py --methods dama --lr 5e-4 --tag _abl$ab
done
ABL=nofreeze python dama_repro.py --methods dama --langs cv:ia cv:kmr fleurs:or_in --lr 5e-4 --tag _ablnofreeze
```

**Tables from the result files:**

```
python table.py > results/main_table.txt       # main table, DAMA vs LoRA per language, zero-shot comparison
python abl.py > results/ablation_table.txt     # ablation comparison over 20 languages
```

## Settings

- Training: batch size 6, 2 epochs, AdamW with weight decay 0, bf16 autocast. `ReduceLROnPlateau` (factor 0.8,
  threshold 0.0025) stands in for the paper's NewBob scheduler. Seeds 0 and 1 (`SEED` environment variable, default 0).
- Learning rates: full fine-tuning 1e-5; LoRA, LoRA-FA, DoRA and DAMA 5e-4; LoRA-XS, VB-LoRA and AdaLoRA 1e-3.
- LoRA: rank 64, alpha 64, dropout 0.1. DoRA: same, no dropout. LoRA-FA: A frozen. LoRA-XS: rank 64, frozen SVD A and B,
  trainable r x r matrix. AdaLoRA: initial rank 48, target 32. VB-LoRA: 90 vectors, top-2, rank 64.
- DAMA: U-shaped rank schedule with thresholds 0.3 and 0.7, high rank 32 and low rank 8 (environment variables
  `RH` and `RL`), alpha 32, SVD initialization in the middle layers, middle-layer A matrices frozen.
- Unseen languages get a new language token (a new embedding row trained through a gradient mask). The 5 FLEURS languages
  Whisper knows use their existing tokens.
- Evaluation: greedy decoding (up to 80 tokens for Common Voice, 120 for FLEURS), WER computed with `jiwer` after NFKC
  normalization, lowercasing and punctuation removal. The scheduler uses a 200-utterance development subset; scores come from the
  full test split.

## Repository layout

```
dama_repro.py        main script: data loading, all 8 methods, training, evaluation
go.sh                smoke test, then the full run
build_manifests.py   builds manifests/*.jsonl from the downloaded data
check_data.py        data sanity check
download_fleurs.py   FLEURS download
table.py, abl.py     result tables from runs2/
runs2/               one JSON per run (442 files): seed 0, seed 1, zero-shot, sweep, capacity, ablations
logs/                training logs of every run
results/             main_table.txt and ablation_table.txt
requirements.txt     pip freeze of the working environment
```

## Notes and known differences from the paper

- Whisper-small (12 decoder layers) instead of Whisper large-v2 (32 layers).
- bf16 instead of fp16, and a plateau scheduler instead of NewBob.
- Learning rates were chosen by us; only LoRA, DAMA, LoRA-XS and VB-LoRA were swept (on 3 languages).
- The `trainable_M` field in the JSON files counts the whole token-embedding matrix for languages with a new token,
  although only the new token's row is trained. `table.py` subtracts it and reports `adapterM`.
- Paper results are not copied here. Our numbers come only from the files in `runs2/`.

## Sources and disclosure

- Base paper: "Adapting Where It Matters: Depth-Aware Adaptation for Efficient Multilingual Speech Recognition in Low-Resource Languages" (Xiao, Holden, Dang; Findings of ACL 2026; method name DAMA).
- Data: CL-MASR Common Voice subset (Zenodo 8065754) and FLEURS (`google/fleurs`).
- Libraries: Hugging Face `transformers` (Whisper), `peft` (AdaLoRA and VB-LoRA), `librosa`, `jiwer`, PyTorch.
- LoRA, LoRA-FA, DoRA, LoRA-XS and DAMA are implemented by us in `dama_repro.py` (class `Ad`); no code was copied from other repositories.
- The code was developed with the help of an AI assistant (Claude, Anthropic).
