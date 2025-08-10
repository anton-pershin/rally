from typing import Optional


class Llm:
    def __init__(
        self,
        url: str,
        authorization: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.url: str = url
        self.authorization: Optional[str] = authorization
        self.model: Optional[str] = model


class LocalLlm(Llm):
    def __init__(self, url: str) -> None:
        super().__init__(
            url=url,
        )


class OpenAiApiLlmWithAuthorization(Llm):
    def __init__(self, url: str, api_key: str, model: str) -> None:
        super().__init__(
            url=url,
            authorization=f"Bearer {api_key}",
            model=model,
        )

