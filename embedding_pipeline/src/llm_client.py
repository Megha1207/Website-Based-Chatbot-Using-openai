import os
from openai import OpenAI
import ollama

class LLMClient:
    def __init__(self, provider: str = "openai"):
        self.provider = provider

        if provider == "openai":
            self.client = OpenAI(api_key="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")

    def generate(self, messages):
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "ollama":
            response = ollama.chat(
                model="tinyllama",
                messages=messages,
                options={"temperature": 0}
            )
            return response["message"]["content"].strip()

        else:
            raise ValueError("Unsupported LLM provider")
