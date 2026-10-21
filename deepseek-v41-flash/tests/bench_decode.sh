#!/usr/bin/env bash
# Quick decode bench: single-stream prose generation, reports tok/s.
# usage: bench_decode.sh [n_tokens] [parallel]
N=${1:-256}; P=${2:-1}
URL=http://127.0.0.1:8000/v1/chat/completions
PROMPT='Write a vivid paragraph about a lighthouse keeper who discovers the sea has begun receding overnight, leaving boats stranded and fish glittering in the moonlight. Describe what he does next.'
gen() {
  local s=$(date +%s.%N)
  local out=$(curl -s -m 600 $URL -H 'Content-Type: application/json' -d "{
    \"model\":\"deepseek-v4.1-flash\",\"stream\":false,
    \"max_tokens\":$N,\"temperature\":0.7,
    \"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json,sys;print(json.dumps('$PROMPT'))")}]
  }")
  local e=$(date +%s.%N)
  local usage=$(echo "$out" | python3 -c "import json,sys
try:
  u=json.load(sys.stdin)['usage']; print(u['completion_tokens'], u['prompt_tokens'])
except Exception: print('PARSE_FAIL', file=sys.stderr); print(-1,-1)")
  local ctok=$(echo $usage | cut -d' ' -f1)
  python3 -c "print(f'{$e-$s:.1f}s  {$ctok} tok  {$ctok/($e-$s):.2f} tok/s')"
}
if [ "$P" -eq 1 ]; then gen; else
  for i in $(seq $P); do gen & done; wait
fi
