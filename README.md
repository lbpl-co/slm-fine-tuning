# Fluento SLM Fine-Tuning

DPO (Direct Preference Optimization) fine-tuning pipeline for Fluento — a voice-to-voice English language tutor powered by a small language model.

## What's in this repo

| File / Folder | Purpose |
|---------------|---------|
| `data/fluento_dpo_500.jsonl` | 500 preference pairs for DPO training |
| `data/eval_prompts_500.csv` | 500 prompts for evaluation |
| `gen_dpo.py` | Script that generated the DPO dataset |
| `dequantize_bnb.py` | Convert bitsandbytes 4-bit checkpoint → fp16 for merging |
| `train_models.md` | Step-by-step: train on Google Colab (T4/A100) and compare vs Gemini |
| `inference.md` | Load saved LoRA from Drive and run inference without retraining |

> **Model weights are not stored here.** Save them to Google Drive or HuggingFace Hub — see the guides below.

---

## Dataset — `fluento_dpo_500.jsonl`

500 synthetic preference pairs. Each record:

```json
{
  "id": 1,
  "dimension": "scaffolding",
  "learner_level": "beginner",
  "prompt": "How do I say 'I am hungry' in English?",
  "chosen": "Let's figure it out together! ...",
  "rejected": "The answer is 'I am hungry'. Please repeat after me."
}
```

### Behavioral dimensions

| Dimension | Count | Chosen does | Rejected does |
|-----------|-------|-------------|---------------|
| scaffolding | 113 | Asks leading questions | Gives the answer directly |
| difficulty_calibration | 100 | Matches complexity to level | Over/under-shoots level |
| error_correction | 98 | Warm recast + explanation | Blunt "Wrong. Correct answer is…" |
| verbosity | 98 | Concise, voice-friendly (1-2 sentences) | Long text-style paragraphs |
| turn_taking | 91 | Responds briefly, invites learner | Monologues without pause |

### Learner level split

| Level | Count |
|-------|-------|
| beginner | 188 |
| intermediate | 194 |
| advanced | 118 |

Regenerate the dataset:
```bash
python3 gen_dpo.py
```

---

## Training on Google Colab

See **[train_models.md](train_models.md)** for the full walkthrough:

1. Install TRL + PEFT (no Unsloth required — avoids the mergekit conflict)
2. Load `Qwen/Qwen2.5-7B-Instruct` with 4-bit quantization
3. Add LoRA adapters and run DPO training (~20-30 min on T4)
4. Compare fine-tuned model vs Gemini Flash 2.5 side-by-side

**Time estimates:**

| GPU | Install + load | DPO training | Total |
|-----|----------------|--------------|-------|
| T4 (free) | ~5 min | ~20-30 min | ~35-45 min |
| A100 (Pro) | ~3 min | ~5-10 min | ~15-20 min |

---

## Inference (no retraining)

See **[inference.md](inference.md)** — loads a saved LoRA adapter from Google Drive in ~2 min.

---

## Quick start

```bash
# Clone
git clone https://github.com/lbpl-co/slm-fine-tuning.git
cd slm-fine-tuning

# (Optional) regenerate the DPO dataset
python3 gen_dpo.py

# Train — open train_models.md and follow the steps in Google Colab
```

---

## Model weights

Model weights are **not committed** to this repo (too large). Save them to:
- **Google Drive** — see Step 9 in `COLAB_GUIDE.md`
- **HuggingFace Hub** — uncomment the `push_to_hub` cell in `COLAB_GUIDE.md`

The `fluento-merged/`, `fluento-merged-fp16/`, and `fluento-mlx-4bit/` directories are in `.gitignore`.
