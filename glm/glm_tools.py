"""Translate GLM's native XML tool calls to OpenAI chat messages and deltas."""
import json
import re
import uuid


def prepare_tools(messages, tools, choice, parallel=True):
    """Normalize JSON arguments for the model's bundled Jinja template."""
    messages = [dict(message) for message in messages]
    for message in messages:
        message["content"] = message.get("content") or ""
        if message["role"] == "developer":
            message["role"] = "system"
        if message["role"] == "tool" and not message.get("tool_call_id"):
            raise ValueError("Tool messages require tool_call_id")
        normalized = []
        for call in message.get("tool_calls") or []:
            call = {**call, "function": dict(call["function"])}
            arguments = call["function"].get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError as error:
                    raise ValueError("Historical tool arguments must be a JSON object") from error
            if not isinstance(arguments, dict):
                raise ValueError("Historical tool arguments must be a JSON object")
            call["function"]["arguments"] = arguments
            normalized.append(call)
        if normalized:
            if message["role"] != "assistant":
                raise ValueError("Only assistant messages can contain tool_calls")
            message["tool_calls"] = normalized
    tools = tools or []
    names = [tool["function"]["name"] for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError("Tool names must be unique")
    forced = None
    if isinstance(choice, dict):
        forced = choice["function"]["name"]
        if forced not in names:
            raise ValueError("tool_choice names a function absent from tools")
        tools = [tool for tool in tools if tool["function"]["name"] == forced]
    elif choice == "required" and not tools:
        raise ValueError("tool_choice required needs at least one tool")
    elif choice == "none":
        tools = []
    instruction = ""
    if choice == "required" or forced:
        instruction = (f"You must call the function {forced} on this turn." if forced
                       else "You must call at least one of the provided functions on this turn.")
    if tools and not parallel:
        instruction += " Call at most one function on this turn."
    if choice == "none":
        instruction = "Do not call functions on this turn. Answer directly."
    if instruction:
        messages.insert(0, {"role": "system", "content": instruction.strip()})
    return messages, tools


def parse_call(block, tools):
    """The GLM template uses arg_key/arg_value pairs, not a JSON call body."""
    name, _, _ = block.partition("<arg_key>")
    name = name.strip()
    function = next((tool["function"] for tool in tools
                     if tool["function"]["name"] == name), None)
    if function is None:
        raise ValueError(f"Model called an unadvertised function: {name}")
    pairs = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
    arguments = {}
    rest = block[len(block) - len(block.lstrip()):]
    rest = rest[len(name):]
    properties = function.get("parameters", {}).get("properties", {})
    position = 0
    for match in pairs.finditer(rest):
        if rest[position:match.start()].strip():
            raise ValueError("Malformed model tool arguments")
        key, raw = match.groups()
        key = key.strip()
        if not key or key in arguments:
            raise ValueError("Empty or repeated model tool argument")
        schema = properties.get(key, {})
        if schema.get("type") == "string":
            value = raw
        else:
            try:
                value = json.loads(raw)
            except ValueError as error:
                if schema.get("type") in ("object", "array", "number", "integer", "boolean", "null"):
                    raise ValueError(f"Invalid JSON value for tool argument {key}") from error
                value = raw
        arguments[key] = value
        position = match.end()
    if rest[position:].strip():
        raise ValueError("Malformed model tool arguments")
    return {"id": "call_" + uuid.uuid4().hex, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


class ToolSplitter:
    """Stream visible text; hold only an unfinished native tool-call block."""
    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"

    def __init__(self, tools, parallel=True):
        self.tools = tools
        self.parallel = parallel
        self.pending = ""
        self.in_call = False
        self.calls = []

    def feed(self, text, final=False, truncated=False):
        self.pending += text
        deltas = []
        while self.pending:
            if self.in_call:
                end = self.pending.find(self.CLOSE)
                if end < 0:
                    break
                call = parse_call(self.pending[:end], self.tools)
                if self.calls and not self.parallel:
                    raise ValueError("Model returned multiple calls with parallel_tool_calls=false")
                self.calls.append(call)
                deltas.append({"tool_calls": [{"index": len(self.calls) - 1, **call}]})
                self.pending = self.pending[end + len(self.CLOSE):]
                self.in_call = False
            else:
                start = self.pending.find(self.OPEN)
                if start >= 0:
                    if start:
                        deltas.append({"content": self.pending[:start]})
                    self.pending = self.pending[start + len(self.OPEN):]
                    self.in_call = True
                    continue
                held = 0
                if not final:
                    for count in range(1, len(self.OPEN)):
                        if self.pending.endswith(self.OPEN[:count]):
                            held = count
                visible = self.pending[:-held] if held else self.pending
                self.pending = self.pending[-held:] if held else ""
                if visible:
                    deltas.append({"content": visible})
                break
        if final and self.in_call:
            if not truncated:
                raise ValueError("Model returned an incomplete tool call")
            # A token limit must not execute an incomplete call or expose its XML.
            self.pending = ""
            self.in_call = False
        return deltas
