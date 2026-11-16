"""Olala channel parser: reasoning / answer / tool calls, one shot and streamed."""

import pytest

from mlx_engine.models.olala.parser import (
    ChannelParser,
    ToolCall,
    parse,
    parse_tool_calls,
    reasoning_is_open,
)

ANSWER = "<|channel_start|>final<|content|>"
TOOLS = "<|channel_start|>tools<|content|>"
END = "<|channel_end|>"
CALLS = (
    '<toolcalls><call id="call_0"><name>get_weather</name>'
    '<arguments>{"city": "Paris"}</arguments></call>'
    '<call id="call_1"><name>get_weather</name>'
    '<arguments>{"city": "Tokyo", "unit": "celsius"}</arguments></call></toolcalls>'
)

# (generated text, reasoning_open, reasoning, content, number of tool calls)
CASES = [
    (
        f"think a < b\nmore{END}{ANSWER}51, Paris.{END}<|im_end|>",
        True,
        "think a < b\nmore",
        "51, Paris.",
        0,
    ),
    (f"{ANSWER}Bonjour.{END}<|im_end|>", False, None, "Bonjour.", 0),
    (f"plan{END}{ANSWER}{END}{TOOLS}{CALLS}{END}<|im_end|>", True, "plan", None, 2),
    (f"{ANSWER}{END}{TOOLS}{CALLS}{END}", False, None, None, 2),
    (
        f"plan{END}{ANSWER}Checking.{END}{TOOLS}{CALLS}{END}",
        True,
        "plan",
        "Checking.",
        2,
    ),
    # stopped by the token limit: inside the reasoning, inside the answer, inside a call
    ("still thinking", True, "still thinking", None, 0),
    (f"plan{END}{ANSWER}The answer is", True, "plan", "The answer is", 0),
    (f"plan{END}{ANSWER}{END}{TOOLS}{CALLS[:60]}", True, "plan", None, 0),
    # text after the end of the message is ignored
    (f"r{END}{ANSWER}a{END}<|im_end|>{ANSWER}ignored{END}", True, "r", "a", 0),
    # a channel opened without closing the previous one
    (f"r{ANSWER}a{END}", True, "r", "a", 0),
    # the final channel written twice (seen with the reasoning disabled): first one kept
    (
        f"{ANSWER}51, Paris.{END}{ANSWER}51 and Paris.{END}",
        False,
        None,
        "51, Paris.",
        0,
    ),
    (f"{ANSWER}51.{END}{ANSWER}51.{END}{TOOLS}{CALLS}{END}", False, None, "51.", 2),
    # look-alikes of a marker stay in the text
    (
        f"if x <| y{END}{ANSWER}a <|channel| b <{END}",
        True,
        "if x <| y",
        "a <|channel| b <",
        0,
    ),
]


@pytest.mark.parametrize("text,is_open,reasoning,content,n_calls", CASES)
def test_parse(text, is_open, reasoning, content, n_calls):
    msg = parse(text, reasoning_open=is_open)
    assert (msg.reasoning, msg.content, len(msg.tool_calls)) == (
        reasoning,
        content,
        n_calls,
    )


@pytest.mark.parametrize("text,is_open,reasoning,content,n_calls", CASES)
@pytest.mark.parametrize("size", [1, 2, 3, 7, 16])
def test_streaming_matches_one_shot(text, is_open, reasoning, content, n_calls, size):
    """Markers split across segments at every offset give the same channels."""
    parser = ChannelParser(reasoning_open=is_open)
    r = c = ""
    for i in range(0, len(text), size):
        d = parser.feed(text[i : i + size])
        r, c = r + d.reasoning, c + d.content
    d = parser.finish()
    r, c = r + d.reasoning, c + d.content
    assert (r.strip() or None, c.strip() or None) == (reasoning, content)
    assert len(parser.tool_calls) == n_calls


def test_streaming_emits_text_as_soon_as_it_is_unambiguous():
    parser = ChannelParser(reasoning_open=True)
    assert parser.feed("abc").reasoning == "abc"
    assert not parser.feed("<|chan")  # could be a marker: held back
    assert parser.feed("ge").reasoning == "<|change"  # it was not
    assert not parser.feed(f"{END}<|channel_start|>fin")
    assert parser.feed("al<|content|>Hi").content == "Hi"


def test_tool_calls():
    calls = parse_tool_calls(CALLS)
    assert calls == [
        ToolCall("call_0", "get_weather", '{"city": "Paris"}'),
        ToolCall("call_1", "get_weather", '{"city": "Tokyo", "unit": "celsius"}'),
    ]
    assert calls[1].arguments_dict() == {"city": "Tokyo", "unit": "celsius"}
    pretty = """<toolcalls>
    <call id="call_1">
        <name>calculator</name>
        <arguments>{"expression": "1 < 2"}</arguments>
    </call>
</toolcalls>"""
    assert parse_tool_calls(pretty) == [
        ToolCall("call_1", "calculator", '{"expression": "1 < 2"}')
    ]


def test_reasoning_is_open():
    head = "<|im_start|>user<|channel_start|>text<|content|>hi<|channel_end|><|im_end|><|im_start|>assistant"
    assert reasoning_is_open(head + "<|channel_start|>analysis<|content|>")
    assert not reasoning_is_open(
        head + "<|channel_start|>analysis<|content|><|channel_end|>"
    )
    assert not reasoning_is_open(head)
