"""
Minimal OpenAI client wrapper shared by both evaluator pipelines.

Replaces the original Modules/dynamic_llm_call.py. The DeepSeek fallback, the
Gemini fallback, and the AWS AppConfig provider switch have all been removed --
this talks to OpenAI only.

The point of this file is that the chains read `response.output_text` and
`response.usage.total_cost`, neither of which exist on a real OpenAI response.
`MockOpenAIResponse` supplies them, so the chains work unmodified.

Both surfaces are provided from one client:
  - `.chat.completions.create` -- used by the short-answer chain. Handles the
    GPT-5 / o-series parameter quirks and the `auto_fail` test hook.
  - `.responses.create` / `.responses.parse` -- used by the essay (compo) chain.
"""

from openai import AsyncOpenAI
from openai.resources.chat import AsyncChat, AsyncCompletions
from openai.resources.responses import AsyncResponses

from evaluator.openai_pricing import AI_PRICING


# --- Models that reject temperature/top_p and rename max_tokens ------------

def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


# --- Response shape the chains expect --------------------------------------

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = AI_PRICING.get(model.lower())
    if not pricing:
        return 0.0
    return ((input_tokens * pricing["input"]) + (output_tokens * pricing["output"])) / 1e6


class MockUsage:
    """Mimics OpenAI's usage object, and adds total_cost.

    Exposes both the chat-completions token names (`prompt_tokens` /
    `completion_tokens`) and the responses-API names (`input_tokens` /
    `output_tokens`) so callers of either surface read the counts they expect.
    """
    def __init__(self, input_tokens: int, output_tokens: int, model: str):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.total_cost = calculate_cost(model, input_tokens, output_tokens)


class MockOpenAIResponse:
    """What the chains actually consume: .output_text, .usage, .model."""
    def __init__(self, text: str, input_tokens: int, output_tokens: int, model: str):
        self.output_text = text
        self.usage = MockUsage(input_tokens, output_tokens, model)
        self.model = model


# --- chat.completions -------------------------------------------------------

class DynamicAsyncCompletions(AsyncCompletions):
    async def create(self, *args, **kwargs):
        extra_body = kwargs.get("extra_body", {}) or {}

        # Test hook preserved from the original: auto_fail=True forces an error
        # so you can exercise the retry loop in the endpoint.
        if extra_body.pop("auto_fail", False):
            raise Exception("Testing for auto fail")
        extra_body.pop("debug_mode", None)          # was the AppConfig switch
        kwargs["extra_body"] = extra_body or None

        # GPT-5 / o-series don't accept these. Translate rather than crash.
        if _is_reasoning_model(kwargs.get("model", "")):
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            kwargs.setdefault("reasoning_effort", "none")

        response = await super().create(*args, **kwargs)

        return MockOpenAIResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=kwargs["model"],
        )


class DynamicAsyncChat(AsyncChat):
    def __init__(self, client: AsyncOpenAI) -> None:
        super().__init__(client)
        self.completions = DynamicAsyncCompletions(client)


# --- responses --------------------------------------------------------------

class DynamicAsyncResponses(AsyncResponses):
    def _wrap(self, response, model):
        return MockOpenAIResponse(
            text=response.output_text,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            model=model,
        )

    async def create(self, *args, **kwargs):
        response = await super().create(*args, **kwargs)
        return self._wrap(response, kwargs["model"])

    async def parse(self, *args, **kwargs):
        # text_format is forwarded untouched -- the SDK turns the Pydantic
        # model into the JSON schema it sends to OpenAI.
        response = await super().parse(*args, **kwargs)
        return self._wrap(response, kwargs["model"])


# --- the client ------------------------------------------------------------

class DynamicAsyncOpenAI(AsyncOpenAI):
    """Drop-in for the original. Same constructor, same call sites."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat = DynamicAsyncChat(self)
        self.responses = DynamicAsyncResponses(self)
