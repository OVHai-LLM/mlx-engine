"""Channel parser for the Olala chat format (reasoning, answer, tool calls).

An assistant turn is a sequence of channels:

    <|channel_start|>analysis<|content|>...<|channel_end|>
    <|channel_start|>final<|content|>...<|channel_end|>
    <|channel_start|>tools<|content|><toolcalls>...</toolcalls><|channel_end|><|im_end|>

The chat template ends the generation prompt inside the analysis channel
(`...assistant<|channel_start|>analysis<|content|>`), so the generated text starts in
the middle of the reasoning, without an opening marker. With the reasoning disabled the
prompt closes an empty analysis channel and the output starts with the final channel.

Same behaviour as the vLLM parsers (olala_reasoning_parser.py / olala_tool_parser.py),
but on text instead of token ids: mlx-engine yields text segments, and a marker can be
split across two segments, so the tail of a segment that could start a marker is held
back until the next one.
"""

import json
import re
from dataclasses import dataclass, field

CHANNEL_START = "<|channel_start|>"
CONTENT = "<|content|>"
CHANNEL_END = "<|channel_end|>"
END_OF_MESSAGE = "<|im_end|>"

REASONING_CHANNEL = "analysis"
ANSWER_CHANNEL = "final"
TOOLS_CHANNEL = "tools"

REASONING_START = f"{CHANNEL_START}{REASONING_CHANNEL}{CONTENT}"

_CALL_RE = re.compile(
    r'<call\s+id="([^"]*)">\s*<name>(.*?)</name>'
    r"\s*<arguments>(.*?)</arguments>\s*</call>",
    re.DOTALL,
)
# markers that end the text of an open channel
_CLOSERS = (CHANNEL_END, CHANNEL_START, END_OF_MESSAGE)


def reasoning_is_open(prompt_text: str) -> bool:
    """Whether a rendered prompt ends inside the analysis channel."""
    return prompt_text.rfind(REASONING_START) > prompt_text.rfind(CHANNEL_END)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    # the arguments as the model wrote them (a JSON object when it is well formed)
    arguments: str

    def arguments_dict(self) -> dict:
        return json.loads(self.arguments)


@dataclass
class ChannelDelta:
    """Text added to each channel by one segment of generated text."""

    reasoning: str = ""
    content: str = ""
    tools: str = ""

    def __bool__(self) -> bool:
        return bool(self.reasoning or self.content or self.tools)


@dataclass
class ParsedMessage:
    reasoning: str | None
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


def parse_tool_calls(tools_text: str) -> list[ToolCall]:
    """The complete <call> elements of a tools channel. A call cut off by the token limit
    is dropped, like in the vLLM tool parser."""
    calls = []
    for call_id, name, arguments in _CALL_RE.findall(tools_text):
        arguments = arguments.strip()
        try:  # normalised JSON text; left as written when the model's JSON is invalid
            arguments = json.dumps(json.loads(arguments), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        calls.append(ToolCall(id=call_id, name=name.strip(), arguments=arguments))
    return calls


def _held_back(text: str, markers) -> int:
    """Length of the longest suffix of text that is a proper prefix of a marker."""
    best = 0
    for m in markers:
        for n in range(min(len(m) - 1, len(text)), best, -1):
            if text.endswith(m[:n]):
                best = n
                break
    return best


class ChannelParser:
    """Incremental parser: feed() generated text segments, then finish()."""

    def __init__(self, reasoning_open: bool = True):
        # the channel being written, None between two channels
        self._channel = REASONING_CHANNEL if reasoning_open else None
        self._pending = ""
        self._ended = False
        self._tools = ""
        self._answered = False  # a final channel has been closed

    def feed(self, text: str) -> ChannelDelta:
        delta = ChannelDelta()
        self._pending += text
        while self._pending and not self._ended:
            if self._channel is None:
                if not self._open_channel(delta):
                    break
            elif not self._write_channel(delta):
                break
        if self._ended:
            self._pending = ""
        return delta

    def finish(self) -> ChannelDelta:
        """Flush what feed() held back (generation stopped in the middle of a channel)."""
        delta = ChannelDelta()
        if self._channel is not None and self._pending and not self._ended:
            self._emit(delta, self._pending)
        self._pending = ""
        self._ended = True
        return delta

    @property
    def tool_calls(self) -> list[ToolCall]:
        return parse_tool_calls(self._tools)

    def _emit(self, delta: ChannelDelta, text: str) -> None:
        if not text:
            return
        if self._channel == REASONING_CHANNEL:
            delta.reasoning += text
        elif self._channel == ANSWER_CHANNEL:
            # With the reasoning disabled the model often writes the final channel twice
            # (the same answer again): only the first one is the answer, as in the vLLM
            # parser's extract_reasoning.
            if not self._answered:
                delta.content += text
        elif self._channel == TOOLS_CHANNEL:
            delta.tools += text
            self._tools += text
        # any other channel is not part of an assistant turn: dropped

    def _write_channel(self, delta: ChannelDelta) -> bool:
        """Consume the text of the open channel. False when more text is needed."""
        buf = self._pending
        hits = [(buf.find(m), m) for m in _CLOSERS if m in buf]
        if not hits:
            keep = _held_back(buf, _CLOSERS)
            self._emit(delta, buf[: len(buf) - keep])
            self._pending = buf[len(buf) - keep :]
            return False
        pos, marker = min(hits)
        self._emit(delta, buf[:pos])
        self._answered = self._answered or self._channel == ANSWER_CHANNEL
        self._channel = None
        if marker == CHANNEL_START:
            # a channel opened without closing the previous one: handled by _open_channel
            self._pending = buf[pos:]
        else:
            self._pending = buf[pos + len(marker) :]
            self._ended = marker == END_OF_MESSAGE
        return True

    def _open_channel(self, delta: ChannelDelta) -> bool:
        """Consume up to the next `<|channel_start|>name<|content|>`. False when more
        text is needed. Text between two channels (whitespace) is dropped."""
        buf = self._pending
        start = buf.find(CHANNEL_START)
        eom = buf.find(END_OF_MESSAGE)
        if eom != -1 and (start == -1 or eom < start):
            self._ended = True
            return True
        if start == -1:
            self._pending = buf[
                len(buf) - _held_back(buf, (CHANNEL_START, END_OF_MESSAGE)) :
            ]
            return False
        name_at = start + len(CHANNEL_START)
        content = buf.find(CONTENT, name_at)
        if content == -1:
            self._pending = buf[start:]
            return False
        self._channel = buf[name_at:content].strip()
        self._pending = buf[content + len(CONTENT) :]
        return True


def parse(model_output: str, reasoning_open: bool = True) -> ParsedMessage:
    """Parse a complete generation. Reasoning and answer are stripped, None when empty."""
    parser = ChannelParser(reasoning_open=reasoning_open)
    first, last = parser.feed(model_output), parser.finish()
    return ParsedMessage(
        reasoning=(first.reasoning + last.reasoning).strip() or None,
        content=(first.content + last.content).strip() or None,
        tool_calls=parser.tool_calls,
    )
