from rally.thinking import (THINKING_REMOVERS, remove_nothing,
                            remove_thinking_qwen3, remove_thinking_qwq)


class TestRemoveNothing:
    def test_returns_input_unchanged(self) -> None:
        text = "This is a normal response without thinking tags."
        assert remove_nothing(text) == text

    def test_preserves_multiline(self) -> None:
        text = "Line 1\nLine 2\nLine 3"
        assert remove_nothing(text) == text


class TestRemoveThinkingQwen3:
    def test_takes_last_line(self) -> None:
        text = "<think>thinking process</think>\nFinal answer"
        assert remove_thinking_qwen3(text) == "Final answer"

    def test_multiple_newlines(self) -> None:
        text = "Line 1\nLine 2\nLine 3"
        assert remove_thinking_qwen3(text) == "Line 3"

    def test_single_line(self) -> None:
        text = "Just one line"
        assert remove_thinking_qwen3(text) == "Just one line"


class TestRemoveThinkingQwq:
    def test_strips_thinking_block(self) -> None:
        text = "<think>Long thinking process here</think>\n\nActual response"
        assert remove_thinking_qwq(text) == "Actual response"

    def test_multiple_think_tags(self) -> None:
        text = "<think>first</think>\nmiddle\n<think>second</think>\nfinal"
        assert remove_thinking_qwq(text) == "final"

    def test_no_think_tag(self) -> None:
        text = "Plain response"
        assert remove_thinking_qwq(text) == "Plain response"

    def test_strips_whitespace(self) -> None:
        text = "<think>thinking</think>\n  spaced response  "
        assert remove_thinking_qwq(text) == "spaced response"


class TestThinkingRemovers:
    def test_qwen25_maps_to_remove_nothing(self) -> None:
        assert THINKING_REMOVERS["qwen2.5"] is remove_nothing

    def test_qwen3_maps_to_remove_thinking_qwen3(self) -> None:
        assert THINKING_REMOVERS["qwen3"] is remove_thinking_qwen3

    def test_qwq_maps_to_remove_thinking_qwq(self) -> None:
        assert THINKING_REMOVERS["qwq"] is remove_thinking_qwq

    def test_contains_exactly_three_entries(self) -> None:
        assert len(THINKING_REMOVERS) == 3
