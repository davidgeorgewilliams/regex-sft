# regex-sft

Supervised fine-tuning of Qwen3-4B on text-to-regex, comparing the **Instruct**
and **Thinking** variants under a shared recipe.

Every training label and every evaluation score comes from **executing** the
model's regex against strings it was never shown. No judge model, no human
labelling, no proxy metric.

---

## Charts

`results/curves.html` — self-contained, no dependencies, opens in any browser.
Training loss, the dev sweep used for checkpoint selection, the free-running
turn-by-turn collapse, and gold-vs-free multi-turn. Hover any point for its
value; every chart has a data table beneath it. Regenerate with
`python scripts/make_charts.py`.

## Headline result

Fine-tuning works, the two variants tie overall, and the difference shows up
somewhere more interesting than raw accuracy.

| model | before tuning | after tuning |
|---|---|---|
| Instruct | 46.9% | **61.9%** (+15.0 pp, paired McNemar p<0.05) |
| Thinking | *not established — see below* | **60.6%** |

### Why the Thinking baseline is blank

The untrained Thinking model **does not finish answering**, so it has no
comparable accuracy number. This is measured, not inferred:

| at a 384-token budget, greedy, same 160 test prompts | truncated | committed answer |
|---|---|---|
| untrained Thinking | **100%** | **0%** |
| fine-tuned Thinking | **0%** | 97.5% |

The stored run reports 1.9% "valid JSON" for the untrained model, which was a
parser flaw rather than three real answers. With no `<|im_end|>` and no closing
`</think>`, the brace scan reached into the model's *reasoning* and extracted
candidate JSON it was still deliberating over — a thinking model restates its
working. Two of the three did not even compile; the one that did accounts for
the entire 0.6% figure. `parse_output` now returns `None` when a generation
never emitted EOS, since an answer the model did not finish is not an answer.
Every fine-tuned run had 0% truncation, so no reported result changes.

Identical budget, identical prompts, identical base weights. Every single
untrained generation ran to the cap without emitting `<|im_end|>`, so only 1.9%
contained a parseable answer. Its 0.6% "accuracy" therefore measures the token
cap, not the model, and is not reported.

### What was and was not established

**Measured.** At 384 tokens, greedy, on all 160 test prompts: 100% truncation,
1.9% valid JSON. Every generation ran to the cap without emitting `<|im_end|>`.

**Measured.** One prompt at 1200 tokens under the card's sampling settings
(`results/untrained_thinking_sample.txt`, reproduce with
`scripts/inspect_base_generation.py`): the model is **not looping and not
producing gibberish** — distinct 8-gram ratio **0.97**, most-repeated phrase
just 2x. It reasons coherently, identifies that it needs the `s` flag, and
derives the exact gold answer `BEGIN(.*?)END` about **five times**, without
ever committing to it. "Wait" appears over a dozen times: self-verification
with no natural stopping point on a task this small.

**Not established.** Whether it terminates at all within the card's
recommended 32,768-token budget. It was never run to that limit. A batch of 4
had not finished after 13 minutes, which puts a full 160-turn baseline at
roughly **10 hours** on this hardware.

**Why it is blank.** This is a scoping decision, not a blocker. Because the
model elaborates rather than loops, a longer run would plausibly terminate and
yield a real number — the compute was simply not spent. Claiming "the untrained
model cannot do this task" would overstate what was measured; the 0.6% figure
measures a token cap and is not reported.

What the evidence does support is narrower: within every budget tested, the
untrained model did not produce a committed answer, while the fine-tuned model
answers in ~73 tokens. Part of what SFT bought was decisiveness. How much of
the accuracy gap that accounts for is **unquantified**.

Tuned Thinking vs tuned Instruct: **−1.2 pp, χ²=0.03 — not significant.** The
disagreement is near-symmetric (20 instruct-only correct, 18 thinking-only), so
they get *different* items right rather than one being stronger.

### Where the reasoning trace actually pays

Multi-turn conversations, same 90 test turns, varying only what goes in the
history:

| | Instruct | Thinking |
|---|---|---|
| gold history (given the correct prior answer) | 66.7% | 76.7% |
| **free-running** (reads its *own* prior answer) | **51.1%** | **72.2%** |
| degradation | **−15.6 pp** | **−4.5 pp** |

Free-running, Thinking beats Instruct by **+21.1 pp (χ²=7.90, p<0.05)**.

Turn by turn, free-running:

| turn | Instruct | Thinking |
|---|---|---|
| 1 | 80.0% | 86.7% |
| 2 | 46.7% | 76.7% |
| 3 | 26.7% | 53.3% |

Instruct collapses 80 → 47 → 27. Thinking degrades gently, 87 → 77 → 53.

**Interpretation.** The trace buys **error recovery**, not raw skill. The
degradation column is the unambiguous evidence: over the same 90 turns, Instruct
loses 15.6 pp when it must read its own answers (p<0.05) while Thinking loses
4.5 pp (ns). Instruct pattern-matches its previous answer forward, so an early
mistake propagates; Thinking re-derives from the instruction each turn.

Whether Thinking is also the weaker *single-shot* model depends on the slice, so
it is worth stating carefully rather than as a slogan:

| items with no history to read | n | Instruct | Thinking |
|---|---|---|---|
| standalone single-turn tasks | 70 | **55.7%** | 40.0% |
| trajectory turn 1 | 30 | 80.0% | **86.7%** |
| both pooled | 100 | **63.0%** | 54.0% |

Instruct leads clearly on the 70 standalone tasks — which are the harder ones,
dominated by bounded-range `validate` concepts — and that slice carries the
pooled figure. On the 30 trajectory openers, which are simpler "establish"
patterns, Thinking is slightly ahead. So "Thinking is the weaker single-shot
model" holds on the harder majority but is not universal, and the error-recovery
claim does not depend on it.

This effect is invisible under gold history (+10 pp, not significant). Running
only the conventional teacher-forced evaluation would have concluded the traces
did nothing.

---

## The task

Three families, all executable:

| family | oracle |
|---|---|
| `validate` | `re.fullmatch` must agree with a positive/negative labelling |
| `extract` | `re.search` group 1 must equal an expected substring |
| `substitute` | `re.sub` with a replacement must produce an expected string |

Each spec shows the model a few **visible** example strings and scores it on a
larger **hidden** set. A pattern that reproduces the visible examples without
honouring the instruction fails the hidden set — giving a directly measurable
overfitting signal, computed in microseconds with no judge.

```
Task: match a semantic version... no component may have a leading zero.
Must match:     '1.2.3', '0.0.1', '10.20.30', '1.0.0-beta.1'
Must not match: '1.2', 'v1.2.3', '1.2.3.4', 'hello'
```

None of the visible negatives has a leading zero. The lazy answer
`\d+\.\d+\.\d+` passes every visible example and fails four hidden ones.

## Dataset

700 conversations, one pretty-printed JSON file each.

| split | files | turns | concepts |
|---|---|---|---|
| train | 499 | 799 | 64 |
| dev | 101 | 161 | 18 |
| test | 100 | 160 | 24 |

- **76 concepts** and **30 multi-turn arcs**, every gold answer verified on both splits
- **70/30** single-turn vs 3-turn trajectories
- Split **by concept**, not by row — variants of a concept are near-duplicates
- Four concepts held out of training entirely, to separate "better at this task"
  from "learned regex reasoning"
- Each arc applies two **moves** (NARROW, BROADEN, REPAIR, RETARGET, FLAG,
  PIVOT, ROLLBACK, COMPOSE, INVERT, GENERALIZE), each appearing exactly 6 times
  across the library and present in all three splits

## Build gates

Every stage refuses to ship on failure. Each of these caught a real bug:

| gate | catches |
|---|---|
| `check_concepts.py` | a gold answer that fails its own test set |
| `check_arcs.py` | same, per trajectory turn |
| `check_coverage.py` | composition: move balance, flags, difficulty × family |
| `check_traces.py` | traces citing strings the model cannot see; missing traps |
| `build_dataset.py` | re-reads every written file and re-verifies it from disk |

Correctness gates alone were not enough. An early draft passed every
correctness check while carrying 21 `NARROW` moves against 1 `ROLLBACK`, and
with `hard` difficulty existing only in the `validate` family — which would have
confounded tier analysis with family. Composition needed its own gate.

## Pipeline

```bash
python scripts/download_models.py          # both Qwen3-4B variants
python scripts/build_specs.py              # verify concept + arc library
python scripts/check_coverage.py           # composition gate
python scripts/build_dataset.py            # data/{train,dev,test}/*.json
python scripts/check_traces.py             # trace quality gate
python scripts/render.py                   # tokenise + loss masks, both variants
python scripts/train.py instruct --epochs 3
python scripts/train.py thinking --epochs 3
python scripts/run_pipeline.py             # dev sweep -> select -> test
python scripts/analyze.py                  # report with CIs + paired tests
```

### Rendering

An n-turn trajectory expands into **n training examples**, one per assistant
turn, with `<think>` stripped from history. This is forced, not chosen: the
Thinking model must emit a trace on every turn, but history must not contain
traces. Instruct is expanded **identically** even though it needn't be — a
different sequence count would confound the comparison.

Qwen3-Thinking's template pre-opens `<think>\n` in the generation prompt, so the
target continues from inside an open block. Emitting `<think>` again would
silently train a doubled tag — invisible in the loss curve.
`data/rendered/mask_*.txt` dumps the decoded token mask for inspection.

### Checkpoint selection

On **dev hidden-pass rate**, never validation loss. The targets are short JSON
blobs whose loss is dominated by boilerplate the model fits immediately.

```
           ep1     ep2     ep3
instruct  51.6 →  62.1 →  60.9    peaked at 2
thinking  47.2 →  58.4 →  62.7    still climbing
```

Instruct's loss was still falling (0.0006 → 0.0003) while its dev accuracy had
already turned over. Selecting on loss would have picked the worse checkpoint.

## Known limitations

1. **Thinking is under-trained.** Its dev curve was still rising at epoch 3
   while Instruct had peaked at 2. Its numbers are a lower bound, so the
   free-running gap is conservative.
2. **Unequal supervision.** Thinking trains on 2.5× the target tokens (the
   traces). Inherent to the treatment, but a gain cannot be cleanly attributed
   to reasoning rather than to more supervision.
3. **Traces are authored, not certified.** Written knowing the answer, so they
   are rationalisation. Guaranteed only: the stated pattern is the verified
   gold, no trace cites a string the model cannot see, every trap is reasoned
   about somewhere. Rejection sampling would give genuinely certified traces.
5. **No accuracy baseline for untrained Thinking — deliberately descoped.**
   "Did fine-tuning improve Thinking's accuracy?" is unanswered; only the
   Instruct improvement (+15.0 pp) is established. The model was not run to its
   recommended 32,768-token budget, so it is **unknown** whether it terminates
   there. It is not looping (8-gram ratio 0.97), so a longer run would
   plausibly succeed — the ~10 hours it needs on this hardware was simply not
   spent. Any future attempt must require **0% truncation** to count; a smaller
   cap would repeat the original error.
6. **Two decode configurations.** Checkpoint comparison uses greedy decoding
   for determinism across the dev sweep; baselines should use each card's
   recommended sampling. Both are defensible for their purpose, but they are
   not interchangeable, and applying the checkpoint-comparison config to a
   baseline is exactly the mistake that produced the invalid 0.6% number.
4. **Per-move results are not reportable.** n=6 per move in test; a 6/6 result
   has a 95% interval of [61, 100]. The move balancing still matters — it keeps
   the *training* distribution even — but answering "which moves does SFT
   improve?" needs ~5–10× more test trajectories.
5. **Single seed.** No variance estimate across runs.

## Hardware

Apple M4 Max, 128 GB, PyTorch MPS, bf16, LoRA r=16 on all seven projections
(0.81% of parameters trainable). Training: ~28 min per variant at ~11 s/step.
Fully offline after the initial weight download.
