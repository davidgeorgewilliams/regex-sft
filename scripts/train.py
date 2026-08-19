"""LoRA SFT for one variant (instruct | thinking) on Apple MPS.

Both variants MUST run with identical seed, LoRA config, schedule and data
order. The trace is the only thing that differs; anything else that differs
confounds the comparison.

BATCHING
Sequences vary from ~120 to ~600 tokens. Padding every batch to the global max
would waste roughly half the compute, so batches are formed from
length-sorted examples and only the batch order is shuffled. Padding is then
bounded by the spread inside a batch rather than across the dataset.

CHECKPOINTS
One adapter per epoch, selected afterwards on dev hidden-pass rate rather than
validation loss. Loss here is dominated by JSON boilerplate -- braces, quotes,
the word "pattern" -- which the model gets right immediately, so validation
loss can drift while task accuracy still improves. With a free verifier there
is no reason to select on a proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RENDERED = ROOT / "data" / "rendered"
CHECKPOINTS = ROOT / "models" / "checkpoints"

MODEL_DIR = {
    "instruct": ROOT / "models" / "original" / "Qwen3-4B-Instruct-2507",
    "thinking": ROOT / "models" / "original" / "Qwen3-4B-Thinking-2507",
}

SEED = 20260819
LORA = dict(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)


def set_seed(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)
    torch.mps.manual_seed(s)


def load_examples(variant: str) -> list[dict]:
    path = RENDERED / f"train_{variant}.jsonl"
    return [json.loads(line) for line in open(path)]


def make_batches(examples: list[dict], batch_size: int, rng: random.Random) -> list[list[dict]]:
    """Length-grouped batching: sort by length, chunk, then shuffle chunk order."""
    order = sorted(examples, key=lambda e: len(e["input_ids"]))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


PAD_MULTIPLE = 64


def collate(batch: list[dict], pad_id: int, device: str) -> dict:
    # Pad up to a multiple of 64 rather than to the exact batch max. Every
    # distinct sequence length is a distinct allocation shape for the MPS
    # caching allocator, and with length-grouped batches almost every step had
    # a new shape -- which fragmented the pool and degraded throughput from
    # ~10s/step to ~55s/step over 80 steps. Bucketing collapses hundreds of
    # shapes into a handful at the cost of a little wasted padding.
    n = max(len(e["input_ids"]) for e in batch)
    n = ((n + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    input_ids, labels, mask = [], [], []
    for e in batch:
        k = n - len(e["input_ids"])
        input_ids.append(e["input_ids"] + [pad_id] * k)
        labels.append(e["labels"] + [-100] * k)          # padding never contributes loss
        mask.append([1] * len(e["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(mask, device=device),
    }


def build_model(variant: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR[variant]), dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**LORA))
    model.to(device)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["instruct", "thinking"])
    # bs=4 measured at 411 tok/s on M4 Max; bs=8 dropped to 188 tok/s, so this
    # is bandwidth-bound rather than occupancy-bound. Effective batch is 16.
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    # Six rather than the conventional three: with 799 examples the peak may
    # come early, and checkpointing every epoch lets dev hidden-pass rate pick
    # the best one afterwards instead of fixing it in advance.
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--benchmark", type=int, default=0,
                    help="run N optimizer steps, report throughput, and exit")
    args = ap.parse_args()

    device = "mps"
    set_seed(SEED)
    rng = random.Random(SEED)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR[args.variant]))
    examples = load_examples(args.variant)
    print(f"variant={args.variant}  examples={len(examples)}  "
          f"tokens={sum(len(e['input_ids']) for e in examples):,}")

    t0 = time.time()
    model = build_model(args.variant, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"loaded in {time.time() - t0:.0f}s  |  trainable {trainable:,} of {total:,} "
          f"({100 * trainable / total:.2f}%)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    batches_per_epoch = math.ceil(len(examples) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs if not args.benchmark else args.benchmark

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    out_dir = CHECKPOINTS / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    step = 0
    micro = 0
    t_start = time.time()
    tokens_seen = 0
    model.train()

    for epoch in range(args.epochs):
        for batch in make_batches(examples, args.batch_size, rng):
            b = collate(batch, tok.pad_token_id, device)
            out = model(**b)
            loss = out.loss
            (loss / args.grad_accum).backward()
            # Drop the logits tensor immediately. At batch 4 x 600 tokens x
            # 152k vocab in bf16 that is ~700MB held alive across backward and
            # the optimizer step for no reason.
            last_loss = loss.detach()
            del out, loss
            tokens_seen += int(b["attention_mask"].sum())
            micro += 1

            if micro % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    torch.mps.empty_cache()

                if step % 5 == 0 or step == 1:
                    torch.mps.synchronize()
                    el = time.time() - t_start
                    print(f"  epoch {epoch + 1} step {step}/{total_steps} "
                          f"loss {last_loss.item():.4f} lr {lr_at(step):.2e} "
                          f"{el / step:.2f}s/step {tokens_seen / el:,.0f} tok/s", flush=True)
                    log.append({"step": step, "epoch": epoch + 1,
                                "loss": last_loss.item(), "s_per_step": el / step})

                if args.benchmark and step >= args.benchmark:
                    torch.mps.synchronize()
                    el = time.time() - t_start
                    print(f"\nBENCHMARK  bs={args.batch_size} accum={args.grad_accum}")
                    print(f"  {el / step:.2f}s per optimizer step, {tokens_seen / el:,.0f} tok/s")
                    full = steps_per_epoch * args.epochs
                    print(f"  projected full run: {full} steps ~= "
                          f"{full * el / step / 60:.1f} min for {args.epochs} epochs")
                    return 0

        ck = out_dir / f"epoch{epoch + 1}"
        model.save_pretrained(str(ck))
        print(f"  saved {ck}")

    (out_dir / "trainlog.json").write_text(json.dumps(
        {"variant": args.variant, "seed": SEED, "lora": LORA,
         "batch_size": args.batch_size, "grad_accum": args.grad_accum,
         "epochs": args.epochs, "lr": args.lr, "log": log}, indent=2) + "\n")
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
