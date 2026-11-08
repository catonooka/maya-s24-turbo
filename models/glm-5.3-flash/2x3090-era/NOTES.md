# GLM-5.3-Flash — 2x3090 era archive (engine v1.0.x-v1.1.20)

The original 2-card campaign artifacts, before the 3-GPU rig and the
`models/<model>/<quant>/` convention. Mixed quants (S24, EXL3-3.05) and engine
versions (v1.1.15 ladder, v1.1.20, v1.1.24 finals). Headline of that era:
59 t/s thinking-on at 262k context — see root README History log.

Contents:
- `exl3/` — early EXL3 research (findings, bottleneck trace, settings)
- `v1.1.15-ladder/` — knob-ladder configs + EXL3-50Tps research + functest
- `glm-v1120/` — v1.1.20 engine bench artifacts
- `v1124-final/` — v1.1.24 finals
- `SPEED-PUSH-2026-10-13.md`, `glm-v1120-REPORT.md` — era reports
- `maya-*.json/.log`, `run-maya-*.sh` — setup-generated configs + launchers
  (paths reference the retired 2-card engine tree)

Current serving quants live in ../exl3-3.05bpw/ and ../maya-s24/.
