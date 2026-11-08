# DeepSeek-V4.1-Flash on the LAN — Full Tutorial

Everything about connecting to the local DeepSeek server from any machine, app, or harness. Written for the rig at `192.168.1.105` (2x RTX 3090, Project Maya engine, DeepSeek-V4.1-Flash 320B, 262k context).

---

## 1. The server

| fact | value |
|---|---|
| Host | `ai` (LAN IP `192.168.1.105`) |
| Port | `8080` |
| Base URL | `http://192.168.1.105:8080/v1` (on the server itself: `http://127.0.0.1:8080/v1`) |
| Model ID | `deepseek-v4.1-flash` |
| API key | none (open on the LAN) — send any string or none |
| Context | 262,144 tokens |
| APIs | OpenAI `/v1/chat/completions`, OpenAI `/v1/responses`, Anthropic `/v1/messages` |
| Vision | yes (images via OpenAI `image_url` format) |
| Web chat | open `http://192.168.1.105:8080/` in any browser |

Start / stop on the server:

```bash
sudo systemctl start maya-ds4    # already running as a systemd unit
sudo systemctl stop maya-ds4
# logs:
tail -f ~/maya-ds4-main/ds4-baseline.log
```

Quick health check from any LAN machine:

```bash
curl -s http://192.168.1.105:8080/v1/status | head -c 200
```

---

## 2. Thinking levels (the main knob)

The model reasons by default. Control it per request with `reasoning_effort`:

| value | behavior |
|---|---|
| `"none"` | thinking OFF, fastest replies |
| `"low"` | brief reasoning |
| `"high"` | default balanced reasoning |
| `"max"` | deepest reasoning |
| `1`–`100` | numeric budget (DeepSeek-style, any integer between low and max) |

Verified scaling on a trap question: none = 0 reasoning chars, 20 = 94, max = 96, all answers correct.

### OpenAI dialect

```bash
curl -s http://192.168.1.105:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "Explain NVLink in 2 sentences."}],
    "max_tokens": 500,
    "temperature": 0.6,
    "reasoning_effort": "high"
  }'
```

Response shape: `choices[0].message.content` = the answer, `choices[0].message.reasoning_content` = the thinking text (may be empty when the model finds the task trivial).

### Anthropic dialect (Claude-style apps)

```bash
curl -s http://192.168.1.105:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "max_tokens": 500,
    "messages": [{"role": "user", "content": "Explain NVLink in 2 sentences."}]
  }'
```

Thinking is controlled Anthropic-style here: `"thinking": {"type": "enabled", "budget_tokens": 4096}` maps automatically (under 2K = low, under 8K = high, else max).

### Streaming

Same endpoints with `"stream": true` — SSE deltas as usual. Add `"stream_options": {"include_usage": true}` on the OpenAI route to get token counts in the final chunk.

---

## 3. Wiring common apps

Any OpenAI-compatible app works with:

- **Base URL**: `http://192.168.1.105:8080/v1`
- **API key**: anything (e.g. `no-key`)
- **Model**: `deepseek-v4.1-flash`

| app | where to set it |
|---|---|
| Open WebUI / LibreChat | Settings → Connections → OpenAI API → base URL + key |
| Continue.dev / Cursor-style | config: `apiBase: http://192.168.1.105:8080/v1`, `model: deepseek-v4.1-flash` |
| Claude Code / Anthropic SDK apps | set `ANTHROPIC_BASE_URL=http://192.168.1.105:8080`, `ANTHROPIC_API_KEY=x`, model `deepseek-v4.1-flash` |
| curl / python `openai` lib | point `base_url` at the address above, `api_key="no-key"` |

Python example:

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.1.105:8080/v1", api_key="no-key")
r = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Write a haiku about PCIe."}],
    max_tokens=200,
    reasoning_effort="low",
)
print(r.choices[0].message.content)
```

---

## 4. Hermes harness (this machine)

Four pre-wired provider tiers in `~/.hermes/config.yaml`, one per thinking level:

| provider | level | use for |
|---|---|---|
| `dsv41` | max | hard problems, deep reasoning |
| `dsv41-high` | high | default agent work |
| `dsv41-low` | low | light thinking, faster |
| `dsv41-fast` | none | plain chat, fastest |

Usage in a session:

```
/model deepseek-v4.1-flash
/provider dsv41-high
```

Or one-shot from the terminal:

```bash
hermes chat -q "summarize this" --provider dsv41-fast -m deepseek-v4.1-flash
```

Each tier pins `temperature 0.6` + its effort in `extra_body`; to change a tier's default edit `custom_providers` in `~/.hermes/config.yaml` (the `reasoning_effort` value accepts the names or any integer 1-100).

Vision through the harness: the server accepts images natively; pin the aux route if you swap main models often (see the wiring notes in the campaign repo).

---

## 5. Performance expectations (measured, quality-gated)

| workload | speed |
|---|---|
| decode, fresh prompt | 26-29 t/s |
| decode, warm cache | 30.8-32.5 t/s |
| prefill @33k tokens | 737 t/s (~45 s) |
| prefill @100k tokens | ~830 t/s (~2 min) |
| thinking tokens | generated at decode speed; long reasoning runs proportionally slower to first answer |

Prompt-heavy session? Flip the engine to the prefill shape first on the server:

```bash
sudo bash ~/maya-ds4-main/flip-tp.sh 2   # prefill shape (1145-1167 t/s @33k, decode ~25-27)
sudo bash ~/maya-ds4-main/flip-tp.sh 1   # decode shape (default)
```

Takes ~5 min to reboot the engine; conversations survive the flip (SLOT_KEEP).

---

## 6. Troubleshooting

| symptom | fix |
|---|---|
| `connection refused` | engine down: `sudo systemctl start maya-ds4` on the server, wait ~5 min for first boot |
| very slow first reply | expert cache warming; second identical ask is 20% faster, repeats reach 32 t/s |
| empty `content`, all text in `reasoning_content` | thinking ate the `max_tokens` budget — raise `max_tokens` or use `reasoning_effort: "none"` |
| answer never starts on a hard question | model is thinking at `max`; watch `reasoning_content` stream or lower the effort |
| app sends `Authorization: Bearer junk` and fails | shouldn't happen — server ignores auth; if a proxy strips the body, bypass it |
| wrong model answers on port 8080 | check which engine owns the port: GLM lives on 8081 when up, DeepSeek on 8080 |

---

*Config, benchmarks and the full campaign log: github.com/catonooka (maya-s24-turbo, `dsv41-3090/deepseek-campaign-1024`).*
