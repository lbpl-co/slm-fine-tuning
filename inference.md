# Fluento Inference Notebook (no retraining needed)

Loads the fine-tuned LoRA adapter from Google Drive. Use this for:
- Testing new prompts
- Generating comparison data for the eval site
- Pushing results to Firestore
- Comparing against Gemini Flash 2.5

**Requirements:** Colab with T4 GPU (free tier works). No TPU needed.
**Load time:** ~2 min (vs ~30 min to retrain)

---

## Cell 1: Install dependencies

```python
!pip install "trl==0.24.0" "transformers>=4.45" "peft>=0.13" "accelerate" "datasets" "bitsandbytes" google-generativeai firebase-admin
```

---

## Cell 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify your model files exist
!ls /content/drive/MyDrive/fluento-dpo-lora/
!ls /content/drive/MyDrive/fluento-merged/
```

---

## Cell 3: Load base model + LoRA adapter (~2 min)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Same base model you trained on
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Path to your saved LoRA adapter on Drive
LORA_PATH = "/content/drive/MyDrive/fluento-dpo-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load LoRA adapter on top
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print(f"Model loaded from {LORA_PATH}")
```

---

## Cell 4: Helper function

```python
SYSTEM = "You are Fluento, a friendly English language tutor for voice conversations. Keep responses short (1-2 sentences), ask guiding questions, and match the learner's level."

def ask_fluento(prompt, max_tokens=150, temperature=0.7):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
```

---

## Cell 5: Quick test

```python
prompts = [
    "I goed to the store yesterday.",
    "What is the past tense of 'go'?",
    "Can you explain what a phrasal verb is?",
    "Between you and I, she's going to resign.",
]

for p in prompts:
    print(f"\nSTUDENT: {p}")
    print(f"FLUENTO: {ask_fluento(p)}")
    print("—" * 50)
```

---

## Cell 6: Configure Gemini Flash 2.5

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")  # <-- paste your key
gemini = genai.GenerativeModel("gemini-2.5-flash")

print("Gemini configured!")
```

---

## Cell 7: Side-by-side comparison

```python
test_prompts = [
    ("beginner",     "What is the past tense of 'go'?"),
    ("beginner",     "I goed to the store yesterday."),
    ("beginner",     "I want to learn English. I am new student."),
    ("intermediate", "Can you explain what a phrasal verb is?"),
    ("intermediate", "She said me that she was coming."),
    ("intermediate", "How do I use 'since' and 'for' with present perfect?"),
    ("advanced",     "I've been learning English for 10 years. Give me something challenging."),
    ("advanced",     "Between you and I, she's going to resign."),
    ("advanced",     "For all intensive purposes, the project is done."),
    # Add more prompts here as needed
]

results = []

for level, prompt in test_prompts:
    fluento_resp = ask_fluento(prompt)
    gemini_resp = gemini.generate_content(
        f"{SYSTEM}\n\nStudent ({level} level): {prompt}"
    ).text.strip()

    results.append((level, prompt, fluento_resp, gemini_resp))

    print(f"\n{'='*70}")
    print(f"[{level.upper()}] STUDENT: {prompt}")
    print(f"\n  FLUENTO (DPO Qwen 7B):")
    print(f"  {fluento_resp}")
    print(f"\n  GEMINI FLASH 2.5:")
    print(f"  {gemini_resp}")
    print(f"\n  Length — Fluento: {len(fluento_resp)} chars | Gemini: {len(gemini_resp)} chars")
```

---

## Cell 8: Summary scorecard

```python
print(f"\n{'='*70}")
print("SUMMARY SCORECARD")
print(f"{'='*70}")
print(f"\n{'Metric':<25} {'Fluento (DPO)':<20} {'Gemini Flash 2.5':<20}")
print(f"{'-'*65}")

fluento_lengths = [len(r[2]) for r in results]
gemini_lengths = [len(r[3]) for r in results]

print(f"{'Avg response length':<25} {sum(fluento_lengths)//len(fluento_lengths):<20} {sum(gemini_lengths)//len(gemini_lengths):<20}")
print(f"{'Max response length':<25} {max(fluento_lengths):<20} {max(gemini_lengths):<20}")
print(f"{'Min response length':<25} {min(fluento_lengths):<20} {min(gemini_lengths):<20}")

# Count responses that end with a question (scaffolding signal)
fluento_questions = sum(1 for r in results if '?' in r[2])
gemini_questions = sum(1 for r in results if '?' in r[3])
print(f"{'Asks a question':<25} {fluento_questions}/{len(results):<17} {gemini_questions}/{len(results):<17}")

# Count short responses (under 200 chars — good for voice)
fluento_short = sum(1 for r in results if len(r[2]) < 200)
gemini_short = sum(1 for r in results if len(r[3]) < 200)
print(f"{'Voice-friendly (<200ch)':<25} {fluento_short}/{len(results):<17} {gemini_short}/{len(results):<17}")

print(f"\nKey things to look for:")
print(f"  - Does Fluento ask leading questions instead of giving answers? (scaffolding)")
print(f"  - Are Fluento responses shorter? (verbosity control for voice)")
print(f"  - Does Fluento encourage the student to try again? (error correction style)")
print(f"  - Does Fluento match the complexity to the level tag? (difficulty calibration)")
```

---

## Cell 9: (Optional) Test with custom prompts

```python
# Interactive testing — add any prompt you want
custom_prompts = [
    "I am boring in this class.",
    "How do I make small talk at a party?",
    "What does 'break the ice' mean?",
    "I wish I can fly.",
    "How do I write a formal email?",
]

for p in custom_prompts:
    fluento_resp = ask_fluento(p)
    gemini_resp = gemini.generate_content(f"{SYSTEM}\n\nStudent: {p}").text.strip()

    print(f"\n{'='*60}")
    print(f"STUDENT: {p}")
    print(f"FLUENTO: {fluento_resp}")
    print(f"GEMINI:  {gemini_resp}")
```

---

## Cell 10: (Optional) Batch test from DPO dataset

```python
# Test the model on a random sample from the training data
# to verify it learned the preferred behaviors
import json, random

with open("/content/drive/MyDrive/fluento_dpo_500.jsonl") as f:
    all_pairs = [json.loads(line) for line in f]

sample = random.sample(all_pairs, 10)

matches = 0
for pair in sample:
    response = ask_fluento(pair["prompt"])

    # Simple heuristic: is the response more similar to chosen or rejected?
    # (by length — chosen should be shorter for verbosity pairs)
    print(f"\n{'='*60}")
    print(f"[{pair['dimension']}|{pair['learner_level']}] {pair['prompt']}")
    print(f"  MODEL:    {response[:120]}")
    print(f"  CHOSEN:   {pair['chosen'][:120]}")
    print(f"  REJECTED: {pair['rejected'][:120]}")
```

---

## Quick Reference

| What | Command |
|------|---------|
| Load time | ~2 min (vs ~30 min to retrain) |
| GPU needed | T4 (free Colab) — no TPU needed |
| Model on Drive | `/content/drive/MyDrive/fluento-dpo-lora/` |
| Merged model | `/content/drive/MyDrive/fluento-merged/` |
| Training data | `/content/drive/MyDrive/fluento_dpo_500.jsonl` |
