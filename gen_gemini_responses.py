"""
Generate Gemini Pro responses for every prompt in the DPO dataset.

Output: data/gemini_pro_responses.jsonl
  Each line: { "id", "dimension", "learner_level", "prompt", "gemini_pro_response" }

Resume-safe: already-processed IDs are skipped on re-run.

Usage:
  pip install google-genai python-dotenv
  python gen_gemini_responses.py  # reads GOOGLE_API_KEY from .env
"""

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "data/fluento_dpo_500.jsonl"
OUTPUT_FILE = "data/gemini_pro_responses.jsonl"
MODEL_NAME   = "gemini-2.5-pro"
PROMPT_FILE  = "prompt.txt"
RATE_LIMIT_DELAY = 1.5   # seconds between requests — adjust if you hit quota

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = open(PROMPT_FILE).read().strip()
print(f"Using prompt from {PROMPT_FILE}:\n{SYSTEM_PROMPT}\n")

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
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_text,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
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
