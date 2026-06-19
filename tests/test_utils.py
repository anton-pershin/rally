import pytest

from rally.utils.common import to_boolean


class TestToBoolean:
    def test_yes_returns_true(self) -> None:
        assert to_boolean("yes") is True

    def test_true_returns_true(self) -> None:
        assert to_boolean("true") is True

    def test_yes_case_insensitive(self) -> None:
        assert to_boolean("Yes") is True
        assert to_boolean("YES") is True
        assert to_boolean("yEs") is True

    def test_true_case_insensitive(self) -> None:
        assert to_boolean("True") is True
        assert to_boolean("TRUE") is True
        assert to_boolean("tRuE") is True

    def test_no_returns_false(self) -> None:
        assert to_boolean("no") is False

    def test_false_returns_false(self) -> None:
        assert to_boolean("false") is False

    def test_no_case_insensitive(self) -> None:
        assert to_boolean("No") is False
        assert to_boolean("NO") is False
        assert to_boolean("nO") is False

    def test_false_case_insensitive(self) -> None:
        assert to_boolean("False") is False
        assert to_boolean("FALSE") is False
        assert to_boolean("fAlSe") is False

    def test_invalid_input_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            to_boolean("maybe")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            to_boolean("")

    def test_unrelated_word_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            to_boolean("hello")
