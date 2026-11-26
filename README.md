# StudyChat

Чат-помощник для учёбы на Streamlit. Всё в одном окне, как в мессенджере: бот сразу задаёт 12 коротких вопросов, потом объясняет темы под тебя. Режимы (объяснить, простыми словами, пример, конспект, проверка, план) переключаются над строкой ввода.

- Любой OpenAI-совместимый API: OpenRouter, OpenAI, Groq, DeepSeek, Gemini, Mistral, Together или свой URL
- Ключ вставляется в меню слева или берётся из Secrets
- RU/EN, история чата по ссылке с `?s=...`

## Запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Онлайн

share.streamlit.io → New app → этот репозиторий, ветка `main`, файл `app.py`. Общий ключ: Advanced settings → Secrets → `OPENROUTER_API_KEY = "..."`.
