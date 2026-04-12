"""
Prompt experimentation loop for tutor response quality.

Workflow:
  1. Edit prompt.txt with your candidate system prompt
  2. Run: python experiment_prompt.py
  3. Review outputs in the terminal
  4. Edit prompt.txt again and re-run
  5. Each run is saved to prompt_experiments.jsonl for comparison

Usage:
  python experiment_prompt.py              # sample 2 prompts per dimension (10 total)
  python experiment_prompt.py --n 3        # 3 per dimension (15 total)
  python experiment_prompt.py --ids 1,5,42 # test specific IDs from the dataset
  python experiment_prompt.py --compare    # print a side-by-side of all saved runs
"""

import argparse
import json
import os
import random
import textwrap
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

DATASET_FILE     = "data/fluento_dpo_500.jsonl"
PROMPT_FILE      = "prompt.txt"
EXPERIMENTS_FILE = "prompt_experiments.jsonl"
DIMENSIONS       = ["scaffolding", "difficulty_calibration", "error_correction", "verbosity", "turn_taking"]
LEVELS           = ["beginner", "intermediate", "advanced"]

# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset():
    with open(DATASET_FILE) as f:
        return [json.loads(line) for line in f]

def sample_rows(rows, n_per_dimension, specific_ids=None):
    if specific_ids:
        id_set = set(specific_ids)
        return [r for r in rows if r["id"] in id_set]

    # Pick n rows per dimension, spread across levels
    selected = []
    by_dim = {d: [r for r in rows if r["dimension"] == d] for d in DIMENSIONS}
    for dim, pool in by_dim.items():
        # Try to get coverage across levels
        by_level = {l: [r for r in pool if r["learner_level"] == l] for l in LEVELS}
        candidates = []
        for level_pool in by_level.values():
            if level_pool:
                candidates.append(random.choice(level_pool))
        # Fill remaining slots randomly from the full dim pool
        while len(candidates) < n_per_dimension:
            candidates.append(random.choice(pool))
        selected.extend(candidates[:n_per_dimension])
    return selected

# ── Run ───────────────────────────────────────────────────────────────────────
def run_experiment(system_prompt, rows):
    from google.genai import types
    results = []

    print(f"\n{'─'*70}")
    print(f"SYSTEM PROMPT:\n{textwrap.indent(system_prompt.strip(), '  ')}")
    print(f"{'─'*70}\n")

    for row in rows:
        user_msg = f"[{row['learner_level']} level] {row['prompt']}"
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=user_msg,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
            generated = response.text.strip()
        except Exception as e:
            generated = f"ERROR: {e}"

        results.append({
            "id":            row["id"],
            "dimension":     row["dimension"],
            "learner_level": row["learner_level"],
            "prompt":        row["prompt"],
            "generated":     generated,
            "chosen":        row["chosen"],   # original hand-crafted response
        })

        # Print
        dim_label = f"{row['dimension']} / {row['learner_level']}"
        print(f"[{dim_label}]")
        print(f"  STUDENT:   {row['prompt']}")
        print(f"  GENERATED: {generated}")
        print(f"  ORIGINAL:  {row['chosen']}")
        print()

    return results

# ── Compare saved runs ────────────────────────────────────────────────────────
def compare_runs():
    if not Path(EXPERIMENTS_FILE).exists():
        print("No experiments saved yet.")
        return

    with open(EXPERIMENTS_FILE) as f:
        runs = [json.loads(line) for line in f]

    if not runs:
        print("No experiments saved yet.")
        return

    print(f"\n{len(runs)} experiment run(s) saved:\n")
    for i, run in enumerate(runs):
        print(f"Run {i+1}  [{run['timestamp']}]  ({len(run['results'])} prompts)")
        print(f"  Prompt: {run['system_prompt'][:120].strip().replace(chr(10), ' ')}...")
        print()

    # Side-by-side for prompts that appear in multiple runs
    # Group by prompt id
    by_id = {}
    for run_idx, run in enumerate(runs):
        for r in run["results"]:
            pid = r["id"]
            if pid not in by_id:
                by_id[pid] = {"prompt": r["prompt"], "learner_level": r["learner_level"],
                               "dimension": r["dimension"], "runs": []}
            by_id[pid]["runs"].append({
                "run": run_idx + 1,
                "generated": r["generated"],
                "timestamp": run["timestamp"],
            })

    overlap = {pid: v for pid, v in by_id.items() if len(v["runs"]) > 1}
    if overlap:
        print(f"{'─'*70}")
        print(f"SIDE-BY-SIDE ({len(overlap)} prompts appear in multiple runs):\n")
        for pid, data in sorted(overlap.items()):
            print(f"[id={pid} | {data['dimension']} / {data['learner_level']}]")
            print(f"  STUDENT: {data['prompt']}")
            for r in data["runs"]:
                print(f"  Run {r['run']}: {r['generated']}")
            print()

# ── Save ──────────────────────────────────────────────────────────────────────
def save_run(system_prompt, results):
    record = {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "system_prompt": system_prompt,
        "results":       results,
    }
    with open(EXPERIMENTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Run saved to {EXPERIMENTS_FILE}  (total runs: {sum(1 for _ in open(EXPERIMENTS_FILE))})")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=2,  help="Samples per dimension (default 2)")
    parser.add_argument("--ids",     type=str, default="", help="Comma-separated dataset IDs to test")
    parser.add_argument("--compare", action="store_true",  help="Print comparison of saved runs")
    args = parser.parse_args()

    if args.compare:
        compare_runs()
        return

    if not Path(PROMPT_FILE).exists():
        raise SystemExit(f"{PROMPT_FILE} not found — create it with your system prompt.")

    system_prompt = Path(PROMPT_FILE).read_text().strip()
    rows = load_dataset()

    specific_ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
    sample = sample_rows(rows, args.n, specific_ids)

    results = run_experiment(system_prompt, sample)
    save_run(system_prompt, results)

if __name__ == "__main__":
    main()
