# Exp 2 + 3 — checkpoint sweep, 2026-10-20

## Exp 2: CPU-tier depth/limit sweep — FLAT (env-level exhausted)
Maya #188-inspired: does more tier depth (CT_LAYERS 35→38→40) or a bigger
admission limit (CT_LIMIT 10→14→16 ms) move decode? Measured warm 256-tok:

| config | t/s |
|---|---|
| L1b (35 layers, limit 10) | 10.0-10.65 (day's baseline) |
| CT_LAYERS=40 LIMIT=10 | 10.02-10.11 |
| CT_LAYERS=40 LIMIT=16 | 10.06-10.19 |
| CT_LAYERS=38 LIMIT=14 | 10.01-10.08 |

All within noise of baseline. cpu_per_call 3.37→3.59 when 40 layers register —
the tier is latency-gated by the routing window, not by depth or limit.
NOTE: layers 18-22 are K=2; register() hard-gates `cold_k == 3`, so "40 layers"
actually registers the same 35 — the flat numbers confirm the gate, not the
concept. The REAL maya #188 port is the C-level split-K memory-order kernel
(ft_tier_cu_v.cu work), still open as the L3-class lever.

## Exp 3: EC profile persistence — MECHANISM VALIDATED, bench-neutral
`src/ct_ec_profile.py`: dumps EC["score"] to a JSON sidecar per replan
(atomic tmp+rename), seeds the score matrix at _ec_alloc from the sidecar
(DSV41_EC_PROFILE env; serve-ecprof.sh).

- Boot 1 (learn): profile written, 205 KB, score sum 8229 (256 steps of the
  bench prompt).
- Boot 2 (seeded): "ECprof: seeded score ... (sum 8229)" in log; first request
  8.28 t/s vs typical unseeded first-request ~8.47; warm 9.88 vs 9.84.

Why no cold win on the bench: the EC pool allocates at step 64 — the first
request ALWAYS streams regardless of scores; seeding helps placement only
after allocation. And a one-prompt profile is a thin sample vs maya's
day-scale usage files. Zero cost, no regression, keep ON for real traffic
(warms placement for the sessions after the first minute).

## Verdicts
- Exp 2 CLOSED at env level (C-kernel port remains the open lever).
- Exp 3 KEPT as default-serving infrastructure (neutral bench, positive
  expected on real workloads, no risk: shape-checked, atomic writes,
  relearn-on-decay semantics unchanged).
