#!/usr/bin/env python3
import json, requests, uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

SGLANG = "http://gpu33:30000/v1/chat/completions"
PORT = 8082

# Only pass these tools to the model — enough for file editing, small enough to fit in context
KEEP_TOOLS = {"Read", "Write", "Edit",
              "read_file", "write_file", "edit_file"}

MAX_SYSTEM_CHARS = 12000  # ~3000 tokens


def anthropic_tools_to_openai(tools):
    filtered = [t for t in tools if t["name"] in KEEP_TOOLS]
    print(f"DEBUG tools: keeping {[t['name'] for t in filtered]} of {[t['name'] for t in tools]}")
    result = []
    for t in filtered:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }
        })
    return result


TOOL_INSTRUCTIONS = """

## Tool Use
To call a tool, respond with ONLY a JSON object in this exact format (no other text):
{"type": "function", "name": "TOOL_NAME", "parameters": {PARAMS}}

Available tools:
- Read(file_path): Read the contents of a file
- Edit(file_path, old_string, new_string, replace_all): Replace text in a file
- Write(file_path, content): Write content to a file

Rules:
- Call one tool per response. Wait for the result before calling another.
- When the task is complete, respond with a short plain-text summary. Do not call any more tools.
- Do not suggest verifying or checking further. Just report what was done and stop.
"""


def anthropic_messages_to_openai(system, messages, has_tools=False):
    result = []
    if system:
        if isinstance(system, list):
            text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
        else:
            text = system
        if len(text) > MAX_SYSTEM_CHARS:
            text = text[:MAX_SYSTEM_CHARS] + "\n[system prompt truncated]"
        if has_tools:
            text += TOOL_INSTRUCTIONS
        result.append({"role": "system", "content": text})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    # Represent tool calls as plain text to avoid sglang tool_calls restriction
                    text_parts.append(json.dumps({
                        "type": "function",
                        "name": block["name"],
                        "parameters": block.get("input", {}),
                    }))
            result.append({"role": "assistant", "content": " ".join(text_parts) or ""})

        elif role == "user":
            text_parts = []
            for block in content:
                if block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = " ".join(b.get("text", "") for b in tool_content)
                    text_parts.append(f"Tool result: {tool_content}")
                elif block.get("type") == "text":
                    text_parts.append(block["text"])
            if text_parts:
                result.append({"role": "user", "content": " ".join(text_parts)})

    return result


def parse_hermes_tool_calls(text):
    import re
    tool_calls = []

    # Format 1: <tool_call>{...}</tool_call> (Hermes)
    pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        for i, m in enumerate(matches):
            try:
                data = json.loads(m)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data.get("name", ""),
                        "arguments": json.dumps(data.get("arguments", data.get("input", {}))),
                    }
                })
            except Exception:
                pass
        clean_text = pattern.sub("", text).strip()
        return tool_calls, clean_text

    # Format 2: JSON tool call embedded anywhere in text (Llama)
    decoder = json.JSONDecoder()
    search = text
    offset = 0
    while True:
        idx = search.find('{"type"')
        if idx == -1:
            break
        try:
            data, end = decoder.raw_decode(search, idx)
            if isinstance(data, dict) and data.get("type") == "function" and "name" in data:
                params = data.get("parameters", data.get("arguments", data.get("input", {})))
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data["name"],
                        "arguments": json.dumps(params),
                    }
                })
                clean_text = (search[:idx] + search[end:]).strip()
                return tool_calls, clean_text
        except Exception:
            pass
        offset += idx + 1
        search = text[offset:]

    return tool_calls, text


def openai_response_to_anthropic(oj, model):
    choice = oj["choices"][0]
    msg = choice["message"]
    usage = oj.get("usage", {})

    content = []
    raw_text = msg.get("content") or ""

    hermes_calls, clean_text = parse_hermes_tool_calls(raw_text)

    if clean_text:
        content.append({"type": "text", "text": clean_text})

    stop_reason = "end_turn"
    all_tool_calls = hermes_calls + (msg.get("tool_calls") or [])
    if all_tool_calls:
        stop_reason = "tool_use"
        for tc in all_tool_calls:
            try:
                inp = json.loads(tc["function"]["arguments"])
            except Exception:
                inp = {}
            # Coerce string booleans to real booleans
            for k in list(inp.keys()):
                if inp[k] == "true":
                    inp[k] = True
                elif inp[k] == "false":
                    inp[k] = False
            content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": inp,
            })

    return {
        "id": "msg_proxy",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args)

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        has_tools = bool(body.get("tools"))
        msgs = anthropic_messages_to_openai(body.get("system"), body.get("messages", []), has_tools)

        if has_tools:
            filtered = [t for t in body["tools"] if t["name"] in KEEP_TOOLS]

        oai_req = {
            "model": "qwen2.5",
            "messages": msgs,
            "max_tokens": min(body.get("max_tokens", 1024), 2048),
        }

        r = requests.post(SGLANG, json=oai_req, timeout=120)
        oj = r.json()
        print(f"sglang response: {json.dumps(oj)[:300]}")

        if "choices" not in oj:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": oj}).encode())
            return

        resp = json.dumps(openai_response_to_anthropic(oj, body.get("model", "qwen2.5"))).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


print(f"Proxy5 listening on port {PORT} -> {SGLANG}")
ThreadedHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
