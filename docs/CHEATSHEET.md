# regex-sft — one-page reference

## Base model — Qwen3-4B (both variants share this)

| | |
|---|---|
| Params | 4.0B total / 3.6B non-embedding |
| Layers | 36 |
| Hidden | 2560 |
| Attention | 32 Q heads, 8 KV heads (GQA) |
| Vocab | 151,936 |
| Context | 262,144 native |
| Precision | bfloat16 |

Variants: **Instruct-2507** (no thinking, never emits `<think>`) and
**Thinking-2507** (always thinks, no toggle). The 2507 release split Qwen3's old
hybrid `enable_thinking` model into two dedicated ones.

## LoRA config — identical for both runs

```
r          = 16
alpha      = 32          (scale = alpha/r = 2)
dropout    = 0.05
bias       = none
targets    = q_proj k_proj v_proj o_proj gate_proj up_proj down_proj   (all 7)
```

**33,030,144 trainable of 4,055,498,240 = 0.81%.** Adapter = 126 MB per checkpoint.

## Optimisation

| | |
|---|---|
| Optimiser | AdamW, betas (0.9, 0.95), weight_decay 0.0 |
| LR | **1e-4**, cosine decay to 0 |
| Warmup | 15 steps (linear) |
| Batch | **4** micro × **4** grad-accum = **effective 16** |
| Grad clip | 1.0 (global norm) |
| Epochs | **3** → 150 optimizer steps |
| Precision | bf16 on Apple MPS, `attn_implementation="eager"` |
| Seed | 20260819 — same data order and init for both runs |

Everything above is byte-identical across the two variants. **The reasoning
trace is the only difference.**

## Data shape

| | train | dev | test |
|---|---|---|---|
| Conversations (files) | 499 | 101 | 100 |
| Turns | 799 | 161 | 160 |
| Concepts | 64 | 18 | 24 |
| Single-turn / 3-turn | 70% / 30% | 70% / 30% | 70% / 30% |

76 concepts total: **30 validate, 23 extract, 23 substitute**. 30 multi-turn
arcs, each applying 2 of 10 conversational moves, each move appearing exactly
6× and present in all three splits. 4 concepts held out of training entirely.

## Rendered sequences (per variant, 799 examples each)

| | Instruct | Thinking |
|---|---|---|
| Seq len median / p95 / max | 250 / 471 / 517 | 295 / 529 / 606 |
| Target tokens mean | 29.4 | 72.9 |
| Target tokens total | 23,461 | **58,225 (2.5×)** |

Padded to multiples of **64** (bucketing — MPS allocator fragments on many
distinct shapes). Loss masked to assistant turns only, built by hand because
Qwen's chat templates lack `{% generation %}` markers.

## Throughput (M4 Max, 128 GB)

```
11 s/step, ~410 tok/s      batch 4   ← chosen
25 s/step, ~190 tok/s      batch 8   ← slower: memory-bandwidth bound
```

Training: **27.5 min** (instruct) / **31.4 min** (thinking). Eval: ~60 min.

## Final training loss

| Instruct | Thinking |
|---|---|
| **0.0003** (memorised) | **0.2678** (trace still carries uncertainty) |

## Checkpoint selection — dev hidden-pass, never val loss

```
           ep1     ep2     ep3
instruct  51.6 →  62.1 →  60.9    peaked → selected epoch 2
thinking  47.2 →  58.4 →  62.7    still rising → selected epoch 3
```

Instruct's loss was still falling (0.0006 → 0.0003) while dev accuracy had
already turned over. Loss-based selection picks the worse checkpoint.

## Evaluation

| | |
|---|---|
| Decode (checkpoints) | greedy, `do_sample=False` — determinism for the sweep |
| Decode (baselines) | model card sampling: Instruct 0.7/0.8/20, Thinking 0.6/0.95/20 |
| max_new_tokens | 128 instruct / 384 thinking |
| Batch | 16, left-padded |
| Metric | **hidden-pass** — pattern executed against strings never shown |
| Multi-turn modes | **gold** (reference history) and **free** (its own answers) |

## Results — test set, n=160

| | hidden pass |
|---|---|
| Instruct base | 46.9% [39–55] |
| **Instruct tuned** | **61.9% [54–69]** — +15.0 pp, χ²=8.53, **p<0.05** |
| **Thinking tuned** | **60.6% [53–68]** — vs Instruct χ²=0.03, **tie** |
| Thinking base | not established (does not terminate) |

**Multi-turn, n=90:**

| | gold history | free-running | drop |
|---|---|---|---|
| Instruct | 66.7% | **51.1%** | −15.6 pp, χ²=9.39 **sig** |
| Thinking | 76.7% | **72.2%** | −4.5 pp, ns |

Free-running gap: **+21.1 pp, χ²=7.90, p<0.05.**

Free-running by turn — Instruct **80 → 47 → 27**, Thinking **87 → 77 → 53**.

Turn 1, no history: Instruct **63.0%** vs Thinking **54.0%** pooled — but that
splits: 55.7 vs 40.0 on the 70 standalone tasks, 80.0 vs 86.7 on the 30
trajectory openers.

## Stats

- **Paired McNemar** — same items both models; only discordant pairs inform
  (20 instruct-only / 18 thinking-only → tie). χ² > 3.84 for p<0.05.
- **Wilson 95% intervals** on every rate.
- **Per-move n=6 → not reported.** 6/6 has a CI of [61, 100].

## Five limitations

1. Thinking under-trained — dev still rising at epoch 3; its number is a floor
2. Thinking trains on 2.5× target tokens — can't separate reasoning from more supervision
3. Traces **authored, not certified** — written knowing the answer, so rationalisation
4. No valid untrained-Thinking baseline — doesn't terminate in an affordable budget
5. Single seed, no variance estimate

## Four bugs found in my own results

1. **384-token cap** → untrained Thinking scored 0.6%, measuring the config not the model
2. **Parser read scratch work** → with no EOS, the brace scan pulled candidate JSON out of the reasoning
3. **21 NARROW vs 1 ROLLBACK** → passed every correctness gate; composition needed its own gate
4. **24% duplicate prompts** → small pools collapsed variants; shrank effective test n
