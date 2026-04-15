# Tutor SLM Fine-Tuning

DPO (Direct Preference Optimization) fine-tuning pipeline for Tutor — a voice-to-voice English language tutor powered by a small language model.

**Live eval site:** https://slm-tutor-finetuning.online/

---

## Repo structure

```
├── data/
│   ├── fluento_dpo_500.jsonl       # 500 DPO preference pairs (training)
│   ├── gemini_pro_responses.jsonl  # Gemini Pro responses for v2 retraining
│   └── eval_prompts_500.csv        # 500 prompts for evaluation
├── docs/
│   ├── train_models.md             # Step-by-step Colab training guide (v1)
│   ├── retrain_with_gemini.md      # Colab guide for Gemini-assisted retraining (v2)
│   └── inference.md                # Load saved LoRA and run inference
├── gen_dpo.py                      # Generates the DPO dataset
├── gen_gemini_responses.py         # Generates Gemini Pro responses for retraining
├── experiment_prompt.py            # Interactive prompt testing loop
└── prompt.txt                      # Active system prompt (edit and re-run to test)
```

> **Model weights are not stored here.** Save them to Google Drive or HuggingFace Hub — see the guides in `docs/`.

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

### v1 — Hand-crafted DPO

See **[docs/train_models.md](docs/train_models.md)** for the full walkthrough:

1. Install TRL + PEFT (no Unsloth required — avoids the mergekit conflict)
2. Load `Qwen/Qwen2.5-7B-Instruct` with 4-bit quantization
3. Add LoRA adapters and run DPO training (~20-30 min on T4)
4. Compare fine-tuned model vs Gemini Flash 2.5 side-by-side

**Time estimates:**

| GPU | Install + load | DPO training | Total |
|-----|----------------|--------------|-------|
| T4 (free) | ~5 min | ~20-30 min | ~35-45 min |
| A100 (Pro) | ~3 min | ~5-10 min | ~15-20 min |

### v2 — Gemini-assisted retraining

See **[docs/retrain_with_gemini.md](docs/retrain_with_gemini.md)**:

1. Generate Gemini Pro responses locally: `python3 gen_gemini_responses.py`
2. Upload both data files to Google Drive
3. Follow the Colab notebook to retrain using Gemini responses as `chosen`

---

## Prompt experimentation

Edit `prompt.txt`, then run:

```bash
python3 experiment_prompt.py              # sample 2 prompts per dimension (10 total)
python3 experiment_prompt.py --n 3        # 3 per dimension (15 total)
python3 experiment_prompt.py --ids 1,5,42 # test specific IDs
python3 experiment_prompt.py --compare    # side-by-side of all saved runs
```

Each run is saved to `prompt_experiments.jsonl` for comparison.

---

## Inference (no retraining)

See **[docs/inference.md](docs/inference.md)** — loads a saved LoRA adapter from Google Drive in ~2 min.

---

## Quick start

```bash
# Clone
git clone https://github.com/lbpl-co/slm-fine-tuning.git
cd slm-fine-tuning

# Set up env
cp .env.example .env  # add your GOOGLE_API_KEY

# (Optional) regenerate the DPO dataset
python3 gen_dpo.py

# (Optional) generate Gemini Pro responses for v2 retraining
python3 gen_gemini_responses.py

# Train — open docs/train_models.md and follow the steps in Google Colab
```

---

## Model weights

Model weights are **not committed** to this repo (too large). Save them to:
- **Google Drive** — see Step 9 in `docs/train_models.md`
- **HuggingFace Hub** — uncomment the `push_to_hub` cell in `docs/train_models.md`

The `fluento-merged/`, `fluento-merged-fp16/`, and `fluento-mlx-4bit/` directories are in `.gitignore`.
