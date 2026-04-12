"""
Generate Gemini Pro responses for every prompt in the DPO dataset.

Output: data/gemini_pro_responses.jsonl
  Each line: { "id", "dimension", "learner_level", "prompt", "gemini_pro_response" }

Resume-safe: already-processed IDs are skipped on re-run.

Usage:
  pip install google-generativeai
  export GEMINI_API_KEY=your_key_here
  python gen_gemini_responses.py
"""

import json
import os
import time

import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/fluento_dpo_500.jsonl"
OUTPUT_FILE = "data/gemini_pro_responses.jsonl"
MODEL_NAME  = "gemini-2.5-pro"
RATE_LIMIT_DELAY = 1.5   # seconds between requests — adjust if you hit quota

SYSTEM_PROMPT = (
    "You are a voice English language tutor. "
    "Your response must be 1-2 sentences maximum — short enough to speak aloud naturally. "
    "Ask a guiding question when appropriate. "
    "Match your vocabulary and complexity to the learner's level. "
    "Never give a lecture. Be warm and encouraging."
)

# ── Setup ─────────────────────────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Set GEMINI_API_KEY environment variable first.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)

# Load already-processed IDs for resume support
done_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    print(f"Resuming — {len(done_ids)} already done.")

# Load input
with open(INPUT_FILE) as f:
    rows = [json.loads(line) for line in f]

todo = [r for r in rows if r["id"] not in done_ids]
print(f"Generating responses for {len(todo)} prompts (of {len(rows)} total)...\n")

# ── Generate ──────────────────────────────────────────────────────────────────
errors = 0

with open(OUTPUT_FILE, "a") as out:
    for i, row in enumerate(todo):
        prompt_text = f"[{row['learner_level']} level] {row['prompt']}"
        try:
            response = model.generate_content(prompt_text)
            result = {
                "id":                  row["id"],
                "dimension":           row["dimension"],
                "learner_level":       row["learner_level"],
                "prompt":              row["prompt"],
                "gemini_pro_response": response.text.strip(),
            }
            out.write(json.dumps(result) + "\n")
            out.flush()

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(todo)}] done={i+1} errors={errors}")

            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            errors += 1
            print(f"  ERROR id={row['id']}: {e}")
            time.sleep(5)

total_done = len(done_ids) + len(todo) - errors
print(f"\nDone. {total_done} responses in {OUTPUT_FILE}  ({errors} errors)")
