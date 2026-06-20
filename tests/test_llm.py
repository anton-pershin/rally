from rally.llm import Llm, LocalLlm, OpenAiApiLlmWithAuthorization


class TestLlm:
    def test_stores_all_attributes(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=10,
            model_family="qwen",
            authorization="Bearer token123",
            model="qwen2.5-7b",
            max_output_tokens=512,
        )

        assert llm.url == "http://localhost:8000"
        assert llm.max_concurrent_requests == 10
        assert llm.model_family == "qwen"
        assert llm.authorization == "Bearer token123"
        assert llm.model == "qwen2.5-7b"
        assert llm.max_output_tokens == 512

    def test_optional_defaults_to_none(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=5,
        )

        assert llm.model_family is None
        assert llm.authorization is None
        assert llm.model is None
        assert llm.max_output_tokens is None


class TestLocalLlm:
    def test_passes_model_family_as_required(self) -> None:
        llm = LocalLlm(
            url="http://localhost:8080",
            max_concurrent_requests=4,
            model_family="llama",
        )

        assert llm.url == "http://localhost:8080"
        assert llm.max_concurrent_requests == 4
        assert llm.model_family == "llama"
        assert llm.max_output_tokens is None

    def test_with_max_output_tokens(self) -> None:
        llm = LocalLlm(
            url="http://localhost:8080",
            max_concurrent_requests=4,
            model_family="llama",
            max_output_tokens=1024,
        )

        assert llm.max_output_tokens == 1024


class TestOpenAiApiLlmWithAuthorization:
    def test_formats_bearer_token(self) -> None:
        llm = OpenAiApiLlmWithAuthorization(
            url="https://api.openai.com/v1/chat/completions",
            max_concurrent_requests=8,
            api_key="sk-test-key",
            model="gpt-4",
        )

        assert llm.authorization == "Bearer sk-test-key"
        assert llm.model == "gpt-4"
        assert llm.max_output_tokens is None

    def test_with_max_output_tokens(self) -> None:
        llm = OpenAiApiLlmWithAuthorization(
            url="https://api.openai.com/v1/chat/completions",
            max_concurrent_requests=8,
            api_key="sk-test-key",
            model="gpt-4",
            max_output_tokens=2048,
        )

        assert llm.max_output_tokens == 2048


class TestEnableThinking:
    def test_enable_thinking_default_none(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=10,
        )

        assert llm.enable_thinking is None

    def test_enable_thinking_set_true(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=10,
            enable_thinking=True,
        )

        assert llm.enable_thinking is True

    def test_enable_thinking_set_false(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=10,
            enable_thinking=False,
        )

        assert llm.enable_thinking is False

    def test_local_llm_enable_thinking(self) -> None:
        llm = LocalLlm(
            url="http://localhost:8080",
            max_concurrent_requests=4,
            model_family="qwen3",
            enable_thinking=False,
        )

        assert llm.enable_thinking is False

    def test_openai_api_llm_enable_thinking(self) -> None:
        llm = OpenAiApiLlmWithAuthorization(
            url="https://api.openai.com/v1/chat/completions",
            max_concurrent_requests=8,
            api_key="sk-test-key",
            model="gpt-4",
            enable_thinking=True,
        )

        assert llm.enable_thinking is True
