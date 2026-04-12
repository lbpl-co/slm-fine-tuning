# Seed Firestore from CSV — Colab Notebook

Runs both Tutor (DPO fine-tuned) and Gemini Flash 2.5 on every prompt in
`eval_prompts_500.csv`, then pushes the results to Firestore as `eval_pairs`.

**Requirements:** Colab with T4 GPU (free tier works). ~45 min for 500 prompts.

---

## Cell 1: Install dependencies

```python
!pip install "trl==0.24.0" "transformers>=4.45" "peft>=0.13" "accelerate" \
             "bitsandbytes" google-generativeai firebase-admin
```

---

## Cell 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify your model files exist
!ls /content/drive/MyDrive/tutor-dpo-lora/
```

---

## Cell 3: Load base model + LoRA adapter (~2 min)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH  = "/content/drive/MyDrive/tutor-dpo-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print(f"Loaded Tutor from {LORA_PATH}")
```

---

## Cell 4: Configure Gemini Flash 2.5

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")  # <-- paste your key
gemini = genai.GenerativeModel("gemini-2.5-flash")

print("Gemini configured!")
```

---

## Cell 5: Configure Firebase

```python
import firebase_admin
from firebase_admin import credentials, firestore

from google.colab import files
uploaded  = files.upload()          # Upload your service account key JSON
KEY_FILE  = list(uploaded.keys())[0]

if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate(KEY_FILE))

db = firestore.client()
print("Firebase connected!")
```

---

## Cell 6: Load CSV from Drive

```python
import csv

CSV_PATH = "/content/drive/MyDrive/eval_prompts_500.csv"  # adjust path if needed

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} prompts")
print(f"Sample: {rows[0]}")
```

> **CSV format expected:**
> ```
> level,prompt
> beginner,What is the past tense of "go"?
> intermediate,Can you explain what a phrasal verb is?
> ...
> ```

---

## Cell 7: Batch run both models and push to Firestore

```python
import hashlib
import time
from firebase_admin import firestore as fs

SYSTEM = (
    "You are Tutor, a friendly English language tutor for voice conversations. "
    "Keep responses short (1-2 sentences), ask guiding questions, and match the learner's level."
)

MODEL_A = "tutor-dpo-v1"   # bump to v2, v3 etc. when you retrain
MODEL_B = "gemini-flash-2.5"

def make_pair_id(prompt, model_a):
    """Deterministic doc ID — prevents duplicates on re-run."""
    raw = f"{prompt.strip().lower()}|{model_a.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]

def ask_tutor(prompt, max_tokens=150):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()

pushed = skipped = errors = 0

for i, row in enumerate(rows):
    level  = row["level"].strip()
    prompt = row["prompt"].strip()

    pair_id = make_pair_id(prompt, MODEL_A)
    doc_ref = db.collection("eval_pairs").document(pair_id)

    if doc_ref.get().exists:
        skipped += 1
        continue

    try:
        tutor_resp = ask_tutor(prompt)
        gemini_resp  = gemini.generate_content(
            f"{SYSTEM}\n\nStudent ({level} level): {prompt}"
        ).text.strip()

        doc_ref.set({
            "prompt":           prompt,
            "level":            level,
            "model_a_name":     MODEL_A,
            "model_a_response": tutor_resp,
            "model_b_name":     MODEL_B,
            "model_b_response": gemini_resp,
            "created_at":       fs.SERVER_TIMESTAMP,
        })
        pushed += 1

        if (pushed + skipped) % 25 == 0:
            print(f"  [{i+1}/{len(rows)}] pushed={pushed} skipped={skipped} errors={errors}")

        time.sleep(0.3)   # gentle rate limit for Gemini API

    except Exception as e:
        errors += 1
        print(f"  ERROR row {i+1} — {e}")
        time.sleep(2)

print(f"\nDone!  pushed={pushed}  skipped={skipped}  errors={errors}")
```

> Re-running this cell is safe — existing docs are skipped via the dedup check.
> To push a new model version, change `MODEL_A = "tutor-dpo-v2"` and re-run.

---

## Cell 8: Verify

```python
docs   = list(db.collection("eval_pairs").stream())
levels = {}
for d in docs:
    lvl = d.to_dict().get("level", "unknown")
    levels[lvl] = levels.get(lvl, 0) + 1

print(f"Total eval pairs in Firestore: {len(docs)}")
for lvl, cnt in sorted(levels.items()):
    print(f"  {lvl}: {cnt}")
```
