from typing import Optional


class Llm:
    def __init__(
        self,
        url: str,
        model_family: Optional[str] = None,
        authorization: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.url: str = url
        self.model_family: Optional[str] = model_family
        self.authorization: Optional[str] = authorization
        self.model: Optional[str] = model


class LocalLlm(Llm):
    def __init__(self, url: str, model_family: str) -> None:
        super().__init__(
            url=url,
            model_family=model_family,
        )


class OpenAiApiLlmWithAuthorization(Llm):
    def __init__(self, url: str, api_key: str, model: str) -> None:
        super().__init__(
            url=url,
            authorization=f"Bearer {api_key}",
            model=model,
        )

