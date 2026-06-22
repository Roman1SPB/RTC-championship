import base64
from openai import OpenAI

ENCODED_KEY = "c2stb3ItdjEtMDA5ZmY3ODMyZTFjMGFkZjYwZjA2M2MxYzlhZTUyNGQyYTI2Y2U0Mzg2NmM0MmU1ZWQ2NDY1YzY5MTRhMTU5MA=="
DECODED_KEY = base64.b64decode(ENCODED_KEY).decode("utf-8")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=DECODED_KEY
)

def get_llm_explanation(user_query: str, code_chunks: list, chunk_names: list) -> str:
    try:
        context_blocks = [f"функция/класс: {name} \n{code}" for name, code in zip(chunk_names, code_chunks)]
        full_context = "\n\n".join(context_blocks)
        
        prompt = f"""Ты — ИИ-ассистент разработчика. Ответь на вопрос по коду проекта.
КОНТЕКСТ: {full_context}
ВОПРОС: {user_query}
ОТВЕТ:"""
        
        response = client.chat.completions.create(
            model="openrouter/owl-alpha",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        if hasattr(response, 'error') and response.error:
            return f"Ошибка от OpenRouter: {response.error.get('message', str(response.error))}"
        
        if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return f"Неожиданный ответ от LLM: {response}"
    except Exception as e:
        return f"Ошибка при обращении к LLM: {str(e)}"