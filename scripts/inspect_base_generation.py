"""Look at what the untrained Thinking model actually generates.

The truncation flag says a generation ran to the cap. It does not say whether
the model was reasoning coherently and simply needed more room, or stuck in a
repetition loop producing nothing of value. Those have completely different
implications -- the first means "give it a bigger budget", the second means
"greedy/sampling settings are broken" -- and the metrics cannot tell them apart.

Single prompt, streamed to stdout, so the answer is visible in ~1 minute
instead of an hour.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "original" / "Qwen3-4B-Thinking-2507"

MAX_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 1200

rec = json.loads(sorted((ROOT / "data" / "test").glob("*.json"))[0].read_text())
turn = rec["turns"][0]

tok = AutoTokenizer.from_pretrained(str(MODEL))
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL), dtype=torch.bfloat16, attn_implementation="eager").to("mps").eval()

msgs = [{"role": "system", "content": rec["system"]},
        {"role": "user", "content": turn["user"]}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

print(f"CONCEPT : {rec['concept']}  ({turn['family']})")
print(f"GOLD    : {turn['gold']['pattern']!r}")
print(f"BUDGET  : {MAX_NEW} new tokens, sampling per model card "
      f"(temp 0.6, top_p 0.95, top_k 20)")
print("=" * 78)

torch.manual_seed(20260819)
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("mps")
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                         temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                         pad_token_id=tok.pad_token_id)

new = out[0][enc["input_ids"].shape[1]:]
text = tok.decode(new, skip_special_tokens=False)

print(text)
print("=" * 78)
print(f"generated tokens : {len(new)}")
print(f"hit the cap      : {len(new) >= MAX_NEW}")
print(f"emitted </think> : {'</think>' in text}")
print(f"emitted <|im_end|>: {'<|im_end|>' in text}")

# Repetition diagnostics: is it looping, or reasoning at length?
words = text.split()
if len(words) > 40:
    grams = [" ".join(words[i:i + 8]) for i in range(len(words) - 8)]
    c = Counter(grams)
    top, n = c.most_common(1)[0]
    uniq = len(c) / len(grams)
    print(f"distinct 8-grams : {uniq:.2f}  (1.0 = no repetition at all)")
    print(f"most repeated    : {n}x  {top[:90]!r}")
    print(f"VERDICT          : {'LOOPING — degenerate repetition' if uniq < 0.5 or n > 5 else 'reasoning at length, not looping'}")
