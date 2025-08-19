from typing import Optional


class Llm:
    def __init__(
        self,
        url: str,
        max_concurrent_requests: int,
        model_family: Optional[str] = None,
        authorization: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.url: str = url
        self.max_concurrent_requests: int = max_concurrent_requests
        self.model_family: Optional[str] = model_family
        self.authorization: Optional[str] = authorization
        self.model: Optional[str] = model


class LocalLlm(Llm):
    def __init__(
        self,
        url: str,
        max_concurrent_requests: int,
        model_family: str
    ) -> None:
        super().__init__(
            url=url,
            max_concurrent_requests=max_concurrent_requests,
            model_family=model_family,
        )


class OpenAiApiLlmWithAuthorization(Llm):
    def __init__(
        self,
        url: str,
        max_concurrent_requests: int,
        api_key: str,
        model: str
    ) -> None:
        super().__init__(
            url=url,
            max_concurrent_requests=max_concurrent_requests,
            authorization=f"Bearer {api_key}",
            model=model,
        )

