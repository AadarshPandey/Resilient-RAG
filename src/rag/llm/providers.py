"""Adapters for LLM providers (Google Gemini, OpenAI, Ollama) with typed error mapping."""

from typing import Any

from rag.config import Settings
from rag.errors import LLMError, LLMRateLimitError, LLMTimeoutError
from rag.llm.base import ChatModel, LLMResult


def _extract_token_usage(response: Any) -> tuple[int, int]:
    """Extract prompt and completion token counts from response metadata."""
    prompt_tokens = 0
    completion_tokens = 0

    # Standard LangChain usage_metadata
    if hasattr(response, "usage_metadata") and isinstance(response.usage_metadata, dict):
        prompt_tokens = int(response.usage_metadata.get("input_tokens", 0))
        completion_tokens = int(response.usage_metadata.get("output_tokens", 0))

    # Gemini style response_metadata
    elif hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
        meta = response.response_metadata
        usage = meta.get("usage_metadata", {})
        prompt_tokens = int(usage.get("prompt_token_count", 0))
        completion_tokens = int(usage.get("candidates_token_count", 0))

    return prompt_tokens, completion_tokens


class GoogleChatModelAdapter:
    """Adapter for Google Generative AI (Gemini) models."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model_name = model_name
        self.client = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
        )

    def invoke(self, messages: list[Any], *, tier: int = 0) -> LLMResult:
        """Invoke Gemini model, mapping provider exceptions to typed errors."""
        try:
            resp = self.client.invoke(messages)
            content = str(resp.content)
            p_tok, c_tok = _extract_token_usage(resp)
            return LLMResult(
                content=content,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                tier=tier,
            )
        except Exception as e:
            err_str = str(e).lower()
            name = type(e).__name__.lower()
            if "resourceexhausted" in name or "429" in err_str or "quota" in err_str:
                raise LLMRateLimitError(f"Gemini rate limit exceeded: {e}") from e
            if "deadline" in name or "504" in err_str or "timed out" in err_str:
                raise LLMTimeoutError(f"Gemini request timed out: {e}") from e
            raise LLMError(f"Gemini API error: {e}", retryable=False) from e


class OpenAIChatModelAdapter:
    """Adapter for OpenAI models."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        self.model_name = model_name
        self.client = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key) if api_key else None,
            temperature=0.0,
        )

    def invoke(self, messages: list[Any], *, tier: int = 0) -> LLMResult:
        """Invoke OpenAI model, mapping provider exceptions to typed errors."""
        try:
            resp = self.client.invoke(messages)
            content = str(resp.content)
            p_tok, c_tok = _extract_token_usage(resp)
            return LLMResult(
                content=content,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                tier=tier,
            )
        except Exception as e:
            err_str = str(e).lower()
            name = type(e).__name__.lower()
            if "ratelimit" in name or "429" in err_str:
                raise LLMRateLimitError(f"OpenAI rate limit: {e}") from e
            if "timeout" in name or "504" in err_str:
                raise LLMTimeoutError(f"OpenAI timeout: {e}") from e
            raise LLMError(f"OpenAI API error: {e}", retryable=False) from e


class OllamaChatModelAdapter:
    """Adapter for local Ollama models."""

    def __init__(self, model_name: str) -> None:
        from langchain_ollama import ChatOllama

        self.model_name = model_name
        self.client = ChatOllama(
            model=model_name,
            temperature=0.0,
        )

    def invoke(self, messages: list[Any], *, tier: int = 0) -> LLMResult:
        """Invoke Ollama model."""
        try:
            resp = self.client.invoke(messages)
            content = str(resp.content)
            p_tok, c_tok = _extract_token_usage(resp)
            return LLMResult(
                content=content,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                tier=tier,
            )
        except Exception as e:
            raise LLMError(f"Ollama execution error: {e}", retryable=False) from e


def build_chat_model(provider: str, model_name: str, settings: Settings) -> ChatModel:
    """Build a ChatModel adapter based on provider name and configuration."""
    prov = provider.lower().strip()
    if prov in ("google", "gemini"):
        api_key = settings.get_effective_gemini_api_key()
        return GoogleChatModelAdapter(model_name=model_name, api_key=api_key)
    elif prov == "openai":
        return OpenAIChatModelAdapter(model_name=model_name, api_key=settings.openai_api_key)
    elif prov == "ollama":
        return OllamaChatModelAdapter(model_name=model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
