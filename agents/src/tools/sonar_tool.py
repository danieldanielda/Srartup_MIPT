from crewai.tools import BaseTool
from openai import OpenAI

from src.settings.config import CrewSettings

settings = CrewSettings()

class SonarSearchTool(BaseTool):
    name: str = "Sonar Barcode Product Lookup"
    description: str = (
        "Use this tool to find the product name, brand, and category by providing a barcode. "
        "Input must be a valid numeric barcode string (e.g., '8809576261752')."
    )

    def _run(self, barcode: str) -> str:
        if not barcode.isdigit():
            return "Product not found."
        
        api_key = settings.model_api_key
        if not api_key:
            return "Error: SONAR_API_KEY not set."

        # Создаём клиент внутри _run
        client = OpenAI(
            api_key=api_key,
            base_url=settings.model_api_base
        )

        prompt = (
            f"Find the exact commercial product name for barcode {barcode}. "
            "Return ONLY the product name in plain text, without any formatting, explanations, or extra words. "
            "If the product is not found, return exactly: Product not found."
        )

        try:
            response = client.chat.completions.create(
                model=settings.model_search_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            result = response.choices[0].message.content.strip()
            if "Product not found" in result:
                return "Product not found."
            return result
        
        except Exception as e:
            return "Product not found."