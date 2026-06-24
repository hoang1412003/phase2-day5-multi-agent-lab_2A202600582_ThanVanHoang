"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.errors import StudentTodoError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion using OpenAI API."""
        from openai import OpenAI
        from multi_agent_research_lab.core.config import get_settings
        
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key) 
        model = settings.openai_model
        
        print(f"[LLMClient] Gọi OpenAI API bằng model '{model}'...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )
        
        content = response.choices[0].message.content or ""
        
        # Lấy thông tin về tokens để tính giá (tùy vào model sẽ có giá khác nhau)
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        
        # Giả định giá của gpt-4o-mini ($0.15/1M input, $0.6/1M output)
        cost_usd = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.6 / 1_000_000)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd
        )
