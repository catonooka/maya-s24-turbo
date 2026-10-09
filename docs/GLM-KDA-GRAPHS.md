# Experimental opt-in KDA layer graphs

`STRATA_GLM_KDA_GRAPH=1` captures recurrent layers of single-device GLM decode.
Default is the ordinary direct path. DSA layers remain direct. Split/MTP,
profiling, dumps, prefetch and lookahead bypass capture. Expert selection, weights,
quantization and kernel arithmetic are unchanged; graph replay explicitly accounts
for MoE requests and residual-buffer swaps. CPU-plan/buffer-key changes invalidate
the corresponding captured graph. Graphs are destroyed before their buffers.

## Evidence and limits

The original implementation was tested on upstream **v1.0.16 / 746ee3f**,
Windows, 4090D 48 GB, Ryzen 5 3600, 128 GB DDR4, Maya-S-v2-IQ2_XXS, 262144
context capacity, six CPU threads and the normal expert pool. Three fresh 32K
thinking-off screening pairs gave medians **30.889 vs 30.114 tok/s (+2.57%)**,
with matching hashes and traffic. Single deep pairs were 29.855 vs 29.086 at
128K and 28.251 vs 27.584 near 258K. These are optimization screens, not a
thinking-on quality or production-readiness claim.

The combined growing-state/quota/KDA candidate is a **different composition**.
Its results must not be attributed to KDA alone. This branch is ported to
**v1.0.21 / 89a1336**; v1.0.16 results are not measured v1.0.21 results.
Keep this PR a draft pending fresh native and thinking-on regression gates on
the port. Synthetic bitwise/abandoned-capture checks and actual-pack full-vocabulary
snapshot/reset comparisons are separate from mixed-plan finite/liveness checks.

## Native validation

Build `glm_layer_graph_test` and `glm_pack_graph_test`. The first is a small
CUDA synthetic test; the second requires the real pack and GPU memory. Use an
isolated `STRATA_GLM_USAGE` and slot directory, MTP off, sufficient RAM tier and
the intended CPU lane. Do not run against live pack sidecars or alongside a
serving engine, and do not mistake compile-only checks for native execution.
The pack test compares the full vocabulary, not only argmax. Preserve any failed
gate; never turn a control's reset/re-prefill difference into a graph parity pass.
