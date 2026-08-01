import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class DeepSeekLLM:
    def __init__(self, model: str = None):
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found. Set it in .env")

        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        print(f"LLM initialized with model: {self.model} (base_url: {self.base_url})")

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM API Error: {e}")
            return ""
