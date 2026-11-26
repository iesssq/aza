"""Клиент для любого OpenAI-совместимого API (OpenRouter, OpenAI, Groq и т.д.)."""
import json
from typing import Iterator

import requests

PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "openrouter/auto",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-001",
            "openai/gpt-4o-mini",
        ],
        "env": "OPENROUTER_API_KEY",
        "keys_url": "https://openrouter.ai/keys",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4.1-mini"],
        "env": "OPENAI_API_KEY",
        "keys_url": "https://platform.openai.com/api-keys",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "env": "GROQ_API_KEY",
        "keys_url": "https://console.groq.com/keys",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "env": "DEEPSEEK_API_KEY",
        "keys_url": "https://platform.deepseek.com/api_keys",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models": ["gemini-2.0-flash", "gemini-1.5-flash"],
        "env": "GEMINI_API_KEY",
        "keys_url": "https://aistudio.google.com/apikey",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "open-mistral-nemo"],
        "env": "MISTRAL_API_KEY",
        "keys_url": "https://console.mistral.ai/api-keys",
    },
    "Together": {
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        "env": "TOGETHER_API_KEY",
        "keys_url": "https://api.together.ai/settings/api-keys",
    },
    "Custom API": {
        "base_url": "",
        "models": [],
        "env": "CUSTOM_API_KEY",
        "keys_url": "",
    },
}


class LLMError(Exception):
    """Ошибка, текст которой не стыдно показать пользователю."""


def _explain_http_error(resp: requests.Response) -> str:
    detail = ""
    try:
        data = resp.json()
        err = data.get("error", data)
        detail = err.get("message", "") if isinstance(err, dict) else str(err)
    except ValueError:
        detail = resp.text[:200]
    hints = {
        401: "Ключ не подошёл. Проверь, что он скопирован целиком и от того провайдера.",
        402: "На счёте не хватает средств для этой модели. Попробуй бесплатную или пополни баланс.",
        403: "Доступ запрещён. Возможно, у ключа нет прав на эту модель.",
        404: "Не нашёл такую модель или адрес. Проверь название модели и Base URL.",
        429: "Слишком много запросов или кончился лимит. Подожди минуту или смени модель.",
    }
    base = hints.get(resp.status_code, f"Сервер ответил ошибкой {resp.status_code}.")
    return f"{base} {detail}".strip()


def stream_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.6,
) -> Iterator[str]:
    """Отдаёт текст ответа по кусочкам."""
    if not base_url:
        raise LLMError("Не указан Base URL.")
    if not model:
        raise LLMError("Не указана модель.")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if "openrouter.ai" in base_url:
        headers["X-Title"] = "Student Helper"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, 120))
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Не получилось достучаться до {base_url}: {e}") from e

    if resp.status_code != 200:
        raise LLMError(_explain_http_error(resp))

    got_any = False
    for raw in resp.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue  # комментарии вроде ": OPENROUTER PROCESSING"
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if "error" in chunk:
            err = chunk["error"]
            raise LLMError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
        for choice in chunk.get("choices", []):
            text = (choice.get("delta") or {}).get("content")
            if text:
                got_any = True
                yield text
    if not got_any:
        raise LLMError("Модель вернула пустой ответ. Попробуй ещё раз или выбери другую модель.")
