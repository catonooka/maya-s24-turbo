"""Tool calls in GLM-5.x's own form (issue #5: every GLM call was "malformed") and Qwen's, streamed and whole, and a
call quoted in a code fence or inline code staying text (Strata #1058).
    python -m unittest serve.test_tools"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser, parse_tool_call  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
        "where": {"type": "object"}, "note": {"type": "string"}}}},
    {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
]
GLM = ("<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>"
       "\n<arg_value>3</arg_value>\n<arg_key>metric</arg_key>\n<arg_value>true</arg_value>\n<arg_key>where</arg_key>"
       "\n<arg_value>{\"lat\": 48.9, \"lon\": 2.35}</arg_value>\n<arg_key>note</arg_key>\n<arg_value>123</arg_value>"
       "\n</tool_call>")
GLM_ARGS = {"city": "Paris", "days": 3, "metric": True, "where": {"lat": 48.9, "lon": 2.35}, "note": "123"}
QWEN = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>"
        "\n</function>\n</tool_call>")


def run(text: str, pieces: int | None = None, tools=TOOLS) -> list:
    """The parser's events for `text` after </think>, fed whole (pieces=None) or `pieces` characters at a time."""
    p = OutputParser(thinking=True, tools=tools, stream_tools=True)
    text = "Let me look.</think>\n\n" + text
    out = []
    step = pieces or len(text)
    for i in range(0, len(text), step):
        out += p.feed(text[i:i + step])
    return out + p.finish()


def calls(events) -> list:
    return [e.call for e in events if e.kind == "tool_call"]


def content(events) -> str:
    return "".join(e.text for e in events if e.kind == "content")


class GlmCalls(unittest.TestCase):
    def test_whole(self):
        c = parse_tool_call(GLM[len("<tool_call>"):-len("</tool_call>")], TOOLS[0])
        self.assertEqual((c.name, c.arguments), ("get_weather", GLM_ARGS))

    def test_without_a_schema_and_without_arguments(self):
        c = parse_tool_call("get_weather<arg_key>days</arg_key><arg_value>3</arg_value>")
        self.assertEqual(c.arguments, {"days": 3})
        self.assertEqual(parse_tool_call("now").arguments, {})

    def test_malformed(self):
        for body in ("<arg_key>a</arg_key><arg_value>1</arg_value>", "f<arg_key>a<arg_value>1</arg_value>",
                     "f junk <b>"):
            with self.assertRaises(ValueError, msg=body):
                parse_tool_call(body)

    def test_streamed_as_whole(self):
        for pieces in (None, 1, 3, 7):
            ev = run("Checking the weather.\n\n" + GLM, pieces)
            got = calls(ev)
            self.assertEqual(len(got), 1, pieces)
            self.assertEqual((got[0].name, got[0].arguments), ("get_weather", GLM_ARGS), pieces)
            starts = [e for e in ev if e.kind == "tool_start"]
            self.assertEqual([s.call.name for s in starts], ["get_weather"], pieces)
            streamed = "".join(e.text for e in ev if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), GLM_ARGS, pieces)       # the pieces add up to the arguments
            self.assertEqual(starts[0].call.id, got[0].id)
            self.assertEqual(content(ev), "Checking the weather.", pieces)

    def test_a_string_value_keeps_its_newlines(self):
        body = "bash<arg_key>command</arg_key><arg_value>\nls -la\n</arg_value>"
        self.assertEqual(parse_tool_call(body, TOOLS[1]).arguments["command"], "\nls -la\n")
        for pieces in (None, 1):
            ev = run("<tool_call>" + body + "</tool_call>", pieces)
            streamed = "".join(e.text for e in ev if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), {"command": "\nls -la\n"}, pieces)

    def test_two_calls(self):
        two = GLM + "\n<tool_call>bash<arg_key>command</arg_key><arg_value>date</arg_value></tool_call>"
        for pieces in (None, 2):
            got = calls(run(two, pieces))
            self.assertEqual([c.name for c in got], ["get_weather", "bash"], pieces)
            self.assertEqual(got[1].arguments, {"command": "date"})

    def test_qwen_form_still_works(self):
        for pieces in (None, 1, 5):
            ev = run(QWEN, pieces)
            got = calls(ev)
            self.assertEqual((got[0].name, got[0].arguments), ("get_weather", {"city": "Paris", "days": 3}), pieces)
            streamed = "".join(e.text for e in ev if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), {"city": "Paris", "days": 3}, pieces)


class QuotedCalls(unittest.TestCase):
    """A call quoted in the answer's code fence or inline code is text; at the top level it is a call."""

    def test_in_a_fence(self):
        for fence in ("```", "~~~", "```xml"):
            text = f"The format looks like this:\n\n{fence}\n" + GLM + "\n" + fence[:3] + "\n\nThat's all."
            for pieces in (None, 1, 4):
                ev = run(text, pieces)
                self.assertEqual(calls(ev), [], (fence, pieces))
                self.assertIn("<tool_call>get_weather", content(ev))
                self.assertTrue(content(ev).endswith("That's all."), (fence, pieces))

    def test_in_inline_code(self):
        text = "Write `<tool_call>bash<arg_key>command</arg_key><arg_value>rm -rf build</arg_value></tool_call>` to run it."
        for pieces in (None, 1, 3):
            ev = run(text, pieces)
            self.assertEqual(calls(ev), [], pieces)
            self.assertEqual(content(ev), text, pieces)

    def test_after_a_closed_fence_and_mid_sentence(self):
        text = "Example:\n```\n<tool_call>x</tool_call>\n```\nNow for real. " + GLM
        for pieces in (None, 1):
            got = calls(run(text, pieces))
            self.assertEqual([c.name for c in got], ["get_weather"], pieces)


class HttpToolCalls(unittest.TestCase):
    """Through the server: a GLM call reaches an OpenAI and an Anthropic client as a tool call."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "Let me look.</think>\n\n" + GLM, max_context=8192), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read())

    def test_openai(self):
        tools = [{"type": "function", "function": t} for t in TOOLS]
        code, b = self.post("/v1/chat/completions", {"model": "m", "tools": tools,
                                                      "messages": [{"role": "user", "content": "weather?"}]})
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        fn = msg["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])), ("get_weather", GLM_ARGS))

    def test_anthropic(self):
        tools = [{"name": t["name"], "input_schema": t["parameters"]} for t in TOOLS]
        code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 4000, "tools": tools,
                                             "messages": [{"role": "user", "content": "weather?"}]})
        self.assertEqual(code, 200, b)
        use = [c for c in b["content"] if c["type"] == "tool_use"]
        self.assertEqual((use[0]["name"], use[0]["input"]), ("get_weather", GLM_ARGS))
        self.assertEqual(b["stop_reason"], "tool_use")


if __name__ == "__main__":
    unittest.main()
