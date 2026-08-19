# Text-to-Regex SFT: Instruct vs Thinking

Scored by execution against hidden strings the model never saw.
Checkpoint chosen on dev hidden-pass rate; test touched once.

## Dev sweep (checkpoint selection)

| variant | ep1 | ep2 | ep3 | selected |
|---|---|---|---|---|
| instruct | 51.6 | 62.1 | 60.9 | **epoch 2** |
| thinking | 47.2 | 58.4 | 62.7 | **epoch 3** |

## Test results (hidden-pass %, gold history)

| metric | base instruct | tuned instruct | base thinking | tuned thinking |
|---|---|---|---|---|
| hidden pass | 46.9 | 61.9 | 0.6 | 60.6 |
| visible pass | 66.9 | 75.6 | 0.6 | 70.6 |
| overfit (vis✓/hid✗) | 20.6 | 15.0 | 0.0 | 11.9 |
| json valid | 98.8 | 100.0 | 1.9 | 97.5 |
| easy | 87.9 | 60.6 | 3.0 | 75.8 |
| medium | 39.1 | 65.5 | 0.0 | 62.7 |
| hard | 17.6 | 41.2 | 0.0 | 17.6 |
| single-turn | 34.3 | 55.7 | 0.0 | 40.0 |
| multi-turn | 56.7 | 66.7 | 1.1 | 76.7 |
| turn 1 | 42.0 | 63.0 | 0.0 | 54.0 |
| turn 2 | 80.0 | 80.0 | 0.0 | 80.0 |
| turn 3 | 30.0 | 40.0 | 3.3 | 63.3 |

## Multi-turn: gold history vs free running

| variant | gold | free |
|---|---|---|
| instruct | 66.7 | 51.1 |
| thinking | 76.7 | 72.2 |

## By conversational move (tuned, gold history)

| move | instruct | thinking |
|---|---|---|
| BROADEN | 0.0 | 66.7 |
| COMPOSE | 100.0 | 100.0 |
| ESTABLISH | 80.0 | 86.7 |
| FLAG | 100.0 | 100.0 |
| GENERALIZE | 100.0 | 100.0 |
| INVERT | 100.0 | 100.0 |
| NARROW | 100.0 | 100.0 |
| PIVOT | 0.0 | 0.0 |
| REPAIR | 0.0 | 100.0 |
| RETARGET | 0.0 | 0.0 |
| ROLLBACK | 100.0 | 50.0 |
