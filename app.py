import json
import os
import re
import uuid
from pathlib import Path

import streamlit as st

from llm import PROVIDERS, LLMError, stream_chat

APP_NAME = "StudyChat"
st.set_page_config(page_title=APP_NAME, page_icon="💬", layout="centered")

DATA_DIR = Path(__file__).parent / "data"
CONTEXT_MESSAGES = 20

# ------------------------------------------------------------------ тексты ---

UI = {
    "ru": {
        "tagline": "Чат-помощник, который объясняет темы так, как удобно именно тебе. "
                   "Сначала выбери, на чём он будет работать, потом ответь на 12 коротких вопросов.",
        "stack_title": "Из чего это сделано",
        "stack": "- **Streamlit** — интерфейс и чат\n"
                 "- **requests** — запросы к API со стримингом ответа\n"
                 "- **OpenRouter, OpenAI, Groq, DeepSeek, Gemini, Mistral, Together** "
                 "или любой свой сервер с OpenAI-совместимым API",
        "step1": "Шаг 1. Модель и ключ",
        "need_key_side": "Вставь ключ в меню слева (стрелка ›) — без него бот не ответит. Вопросы можно проходить уже сейчас.",
        "next": "Дальше — вопросы",
        "q_of": "Вопрос {i} из {n}",
        "skip": "Пропустить",
        "answer_ph": "Напиши ответ или выбери вариант выше",
        "provider": "Провайдер",
        "model": "Модель",
        "model_ph": "например, gpt-4o-mini",
        "base_help": "Любой сервер с OpenAI-совместимым /chat/completions",
        "key": "API-ключ",
        "key_help": "Хранится только в памяти этой сессии и нигде не сохраняется.",
        "get_key": "Взять ключ: {url}",
        "temp": "Температура",
        "temp_help": "Ниже — точнее, выше — свободнее.",
        "profile": "Профиль",
        "model_key": "Модель и ключ",
        "clear": "Очистить чат",
        "restart": "Пройти вопросы заново",
        "retry": "Повторить",
        "mode_label": "Что сделать с темой",
        "chat_ph": "Тема или задание. Режим: {mode}",
        "history_note": "Чат сохраняется: ссылка с `?s=...` вернёт его после обновления страницы.",
        "done": "Готово",
        "remembered": "Запомнил",
        "hard": "Сложнее всего идёт",
        "urgent": "Срочное",
        "welcome_tail": "Напиши тему или вставь задание, а режим вверху подскажет, как с этим работать: "
                        "объяснить, дать пример, сделать конспект или проверить тебя.",
    },
    "en": {
        "tagline": "A chat helper that explains topics the way that works for you. "
                   "First pick what it runs on, then answer 12 short questions.",
        "stack_title": "What it's built with",
        "stack": "- **Streamlit** — UI and chat\n"
                 "- **requests** — streaming API calls\n"
                 "- **OpenRouter, OpenAI, Groq, DeepSeek, Gemini, Mistral, Together** "
                 "or any server with an OpenAI-compatible API",
        "step1": "Step 1. Model and key",
        "need_key_side": "Paste your key in the left menu (arrow ›) — the bot can't answer without it. You can answer the questions right now.",
        "next": "Next — questions",
        "q_of": "Question {i} of {n}",
        "skip": "Skip",
        "answer_ph": "Type an answer or pick an option above",
        "provider": "Provider",
        "model": "Model",
        "model_ph": "e.g. gpt-4o-mini",
        "base_help": "Any server with an OpenAI-compatible /chat/completions",
        "key": "API key",
        "key_help": "Kept only in this session's memory, never saved.",
        "get_key": "Get a key: {url}",
        "temp": "Temperature",
        "temp_help": "Lower is more precise, higher is freer.",
        "profile": "Profile",
        "model_key": "Model and key",
        "clear": "Clear chat",
        "restart": "Redo the questions",
        "retry": "Retry",
        "mode_label": "What to do with the topic",
        "chat_ph": "Topic or task. Mode: {mode}",
        "history_note": "Chat is saved: the link with `?s=...` brings it back after a refresh.",
        "done": "Done",
        "remembered": "Got it",
        "hard": "Hardest right now",
        "urgent": "Urgent",
        "welcome_tail": "Send a topic or paste a task. The mode above picks how I work with it: "
                        "explain, give an example, make notes or quiz you.",
    },
}

# (ключ, текст вопроса, варианты ответа) для каждого языка
QUESTIONS = {
    "ru": [
        ("name", "Привет! Я помощник по учёбе. Для начала — как тебя зовут?", []),
        ("level", "Где учишься?", ["Школа", "Колледж", "Университет", "Учусь сам(а)"]),
        ("field", "Что именно изучаешь? Класс, курс или специальность — как тебе удобно.", []),
        ("hard", "Какие предметы сейчас даются тяжелее всего?", []),
        ("urgent", "Что горит прямо сейчас? Экзамен, сдача, контрольная — напиши с датой, если она есть.",
         ["Пока ничего срочного"]),
        ("style", "Как тебе проще всего понимать новое?",
         ["На примерах", "По шагам", "Через схемы и аналогии", "Коротко и по делу"]),
        ("depth", "Насколько подробно объяснять?", ["Коротко", "Средне", "Подробно, со всеми нюансами"]),
        ("time", "Сколько времени в день ты реально готов(а) тратить на учёбу?",
         ["До 30 минут", "1–2 часа", "3 часа и больше"]),
        ("peak", "Когда лучше соображаешь?", ["Утром", "Днём", "Вечером", "Ночью"]),
        ("blocker", "Что чаще всего мешает?",
         ["Откладываю на потом", "Не могу сосредоточиться", "Не знаю, с чего начать", "Быстро устаю"]),
        ("help", "С чем помогать в первую очередь?",
         ["Разбирать темы", "Решать задачи", "Писать рефераты и тексты", "Код", "Готовиться к экзаменам"]),
        ("tone", "Последний вопрос. Как со мной общаться?",
         ["На «ты», по-дружески", "Спокойно и по делу", "С юмором", "Строго, без поблажек"]),
    ],
    "en": [
        ("name", "Hi! I'm your study helper. First things first — what's your name?", []),
        ("level", "Where do you study?", ["School", "College", "University", "Self-taught"]),
        ("field", "What exactly are you studying? Grade, year or major — whatever fits.", []),
        ("hard", "Which subjects are the hardest for you right now?", []),
        ("urgent", "What's burning right now? Exam, deadline, test — add the date if you have one.",
         ["Nothing urgent yet"]),
        ("style", "How do you understand new things best?",
         ["With examples", "Step by step", "Diagrams and analogies", "Short and to the point"]),
        ("depth", "How detailed should explanations be?", ["Short", "Medium", "Detailed, with all the nuances"]),
        ("time", "How much time a day can you really spend studying?",
         ["Up to 30 minutes", "1–2 hours", "3 hours or more"]),
        ("peak", "When do you think best?", ["Morning", "Afternoon", "Evening", "Night"]),
        ("blocker", "What gets in your way most?",
         ["I put things off", "I can't focus", "I don't know where to start", "I get tired fast"]),
        ("help", "What should I help with first?",
         ["Explaining topics", "Solving problems", "Essays and texts", "Code", "Exam prep"]),
        ("tone", "Last one. How should I talk to you?",
         ["Friendly, casual", "Calm and to the point", "With humor", "Strict, no excuses"]),
    ],
}
NO_URGENT = {QUESTIONS[l][4][2][0] for l in QUESTIONS}

LABELS = {
    "ru": {"name": "Имя", "level": "Где учится", "field": "Что изучает", "hard": "Сложные предметы",
           "urgent": "Срочное", "style": "Как лучше понимает", "depth": "Подробность",
           "time": "Время в день", "peak": "Лучшее время", "blocker": "Что мешает",
           "help": "Чем помогать", "tone": "Тон общения"},
    "en": {"name": "Name", "level": "Studies at", "field": "Studies", "hard": "Hard subjects",
           "urgent": "Urgent", "style": "Learns best by", "depth": "Detail level",
           "time": "Time per day", "peak": "Best time", "blocker": "Gets in the way",
           "help": "Help with", "tone": "Tone"},
}

# id режима -> (подпись, инструкция для модели)
MODES = {
    "explain": {"ru": "📖 Объяснить", "en": "📖 Explain",
                "prompt": "Explain the topic: first the core idea in 2-3 sentences, then a step-by-step breakdown."},
    "simple": {"ru": "🧒 Простыми словами", "en": "🧒 In simple words",
               "prompt": "Explain as simply as possible, as to someone who sees the topic for the first time. Use everyday analogies."},
    "example": {"ru": "🧩 С примером", "en": "🧩 With an example",
                "prompt": "Show the topic through concrete worked examples, explaining each step."},
    "notes": {"ru": "📝 Конспект", "en": "📝 Notes",
              "prompt": "Write short notes: key concepts, formulas or rules, what to remember."},
    "quiz": {"ru": "✅ Проверь меня", "en": "✅ Quiz me",
             "prompt": "Do not explain right away. Ask me 3 questions on the topic one at a time, wait for my answer, then tell me where I was wrong and why."},
    "plan": {"ru": "🗓 План", "en": "🗓 Study plan",
             "prompt": "Make a study plan for the topic, split by days and fitted to my free time."},
}

FOLLOWUPS = {
    "ru": [("Объясни проще", "Объясни то же самое проще, на бытовом примере."),
           ("Ещё пример", "Приведи ещё один пример, другой по типу."),
           ("Глубже", "Разбери подробнее, с нюансами и типичными ошибками."),
           ("Проверь меня", "Проверь меня по этой теме: задай один вопрос и дождись ответа.")],
    "en": [("Simpler", "Explain the same thing more simply, with an everyday example."),
           ("Another example", "Give another example of a different kind."),
           ("Deeper", "Go into more detail, with nuances and common mistakes."),
           ("Quiz me", "Quiz me on this topic: ask one question and wait for my answer.")],
}


def t(key: str, **kw) -> str:
    s = UI[st.session_state.lang][key]
    return s.format(**kw) if kw else s


# ------------------------------------------------------------ сохранение ---

SAVED_KEYS = ("stage", "q", "profile", "survey_log", "chat", "lang")


def valid_sid(s) -> bool:
    return isinstance(s, str) and re.fullmatch(r"[0-9a-f]{32}", s) is not None


def save_state():
    ss = st.session_state
    try:
        DATA_DIR.mkdir(exist_ok=True)
        data = {k: ss[k] for k in SAVED_KEYS}
        cfg = ss.cfg  # ключ доступа намеренно не сохраняем
        data["cfg"] = {k: cfg[k] for k in ("provider", "model", "base_url", "temperature")}
        (DATA_DIR / f"{ss.sid}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # на read-only хостингах просто живём без истории


def load_state(sid: str):
    try:
        data = json.loads((DATA_DIR / f"{sid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    ss = st.session_state
    for k in SAVED_KEYS:
        if k in data:
            ss[k] = data[k]
    if isinstance(data.get("cfg"), dict):
        ss.cfg.update(data["cfg"])


def init_state():
    ss = st.session_state
    if "sid" in ss:
        return
    ss.lang = "ru"
    ss.stage = "chat"
    ss.q = 0
    ss.profile = {}
    ss.survey_log = []  # [(role, text)]
    ss.chat = []  # [{"role", "content"}]
    ss.pending = False
    ss.error = None
    ss.cfg = {"provider": "OpenRouter", "model": PROVIDERS["OpenRouter"]["models"][0],
              "base_url": "", "keys": {}, "temperature": 0.6}
    sid = st.query_params.get("s")
    if not valid_sid(sid):
        sid = uuid.uuid4().hex
        st.query_params["s"] = sid
    ss.sid = sid
    load_state(sid)


def secret_key(provider: str) -> str:
    name = PROVIDERS[provider]["env"]
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


def current_api():
    cfg = st.session_state.cfg
    p = cfg["provider"]
    base = cfg["base_url"] if p == "Custom API" else PROVIDERS[p]["base_url"]
    key = cfg["keys"].get(p) or secret_key(p)
    return base, key, cfg["model"], cfg["temperature"]


# ---------------------------------------------------------- настройки API ---

def on_provider_change():
    ss = st.session_state
    p = ss["w_provider"]
    ss.cfg["provider"] = p
    models = PROVIDERS[p]["models"]
    ss["w_model"] = models[0] if models else ""
    ss["w_key"] = ss.cfg["keys"].get(p) or secret_key(p)


def set_model(name: str):
    st.session_state["w_model"] = name


def api_settings():
    """Одни и те же поля в начале (setup) и в сайдбаре (потом)."""
    ss = st.session_state
    cfg = ss.cfg
    # Виджет переезжает из центра в сайдбар, поэтому при смене места
    # заново подставляем значения из cfg.
    loc = "side"
    if ss.get("_api_loc") != loc:
        ss["w_provider"] = cfg["provider"]
        ss["w_model"] = cfg["model"]
        ss["w_base"] = cfg["base_url"]
        ss["w_key"] = cfg["keys"].get(cfg["provider"]) or secret_key(cfg["provider"])
        ss["w_temp"] = cfg["temperature"]
        ss["_api_loc"] = loc

    st.selectbox(t("provider"), list(PROVIDERS), key="w_provider", on_change=on_provider_change)
    provider = ss["w_provider"]
    info = PROVIDERS[provider]

    if provider == "Custom API":
        st.text_input("Base URL", key="w_base", placeholder="https://my-server.com/v1", help=t("base_help"))
    st.text_input(t("model"), key="w_model", placeholder=t("model_ph"))
    if info["models"]:
        cols = st.columns(min(len(info["models"]), 2))
        for i, m in enumerate(info["models"]):
            cols[i % len(cols)].button(m, key=f"sug_{provider}_{i}", on_click=set_model, args=(m,),
                                       use_container_width=True)
    st.text_input(t("key"), key="w_key", type="password", help=t("key_help"))
    if info["keys_url"]:
        st.caption(t("get_key", url=info["keys_url"]))
    st.slider(t("temp"), 0.0, 1.5, key="w_temp", help=t("temp_help"))

    cfg["provider"] = provider
    cfg["model"] = ss["w_model"].strip()
    cfg["base_url"] = ss["w_base"].strip()
    cfg["keys"][provider] = ss["w_key"].strip()
    cfg["temperature"] = ss["w_temp"]


# ------------------------------------------------------------------ экраны ---

def on_lang_change():
    v = st.session_state["w_lang"]
    if v:
        st.session_state.lang = v


def answer_survey(text: str):
    ss = st.session_state
    key, question, _ = QUESTIONS[ss.lang][ss.q]
    if key == "urgent" and text in NO_URGENT:
        text = ""
    ss.survey_log.append(("assistant", question))
    ss.survey_log.append(("user", text or "—"))
    ss.profile[key] = text
    ss.q += 1
    if ss.q >= len(QUESTIONS[ss.lang]):
        ss.stage = "chat"
        ss.chat.append({"role": "assistant", "content": welcome_text()})


def welcome_text() -> str:
    p = st.session_state.profile
    name = f", {p['name']}" if p.get("name") else ""
    parts = [f"{t('done')}{name}."]
    if p.get("level") or p.get("field"):
        parts.append(f"{t('remembered')}: {p.get('level', '')} {p.get('field', '')}".strip() + ".")
    if p.get("hard"):
        parts.append(f"{t('hard')}: {p['hard']}.")
    if p.get("urgent"):
        parts.append(f"{t('urgent')}: {p['urgent']}.")
    parts.append(t("welcome_tail"))
    return " ".join(parts)


def mode_label(mode_id: str) -> str:
    return MODES[mode_id][st.session_state.lang]


def system_prompt(mode_id: str) -> str:
    ss = st.session_state
    labels = LABELS[ss.lang]
    lines = "\n".join(f"- {labels[k]}: {v}" for k, v in ss.profile.items() if v)
    answer_lang = "Russian" if ss.lang == "ru" else "English"
    return (
        f"You are a personal study helper for a student. Reply in {answer_lang} unless asked otherwise. "
        "Explain so that a groupmate who is struggling would understand: simply, without jargon, "
        "and define any term you use. Do not invent facts: if unsure, say so. No filler introductions.\n\n"
        f"What is known about the student:\n{lines}\n\n"
        "Adapt tone, detail level and format to these answers.\n\n"
        f"Current mode: {MODES[mode_id]['en']}\n{MODES[mode_id]['prompt']}"
    )


def queue_user_message(text: str):
    ss = st.session_state
    ss.chat.append({"role": "user", "content": text})
    ss.pending = True
    ss.error = None


def reset_all():
    keep = (st.session_state.sid, st.session_state.lang, st.session_state.cfg)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    ss = st.session_state
    ss.sid, ss.lang, ss.cfg = keep
    ss.stage, ss.q, ss.profile, ss.survey_log, ss.chat = "chat", 0, {}, [], []
    ss.pending, ss.error = False, None


CSS = """
<style>
.stApp{background:radial-gradient(1200px 600px at 80% -10%,#12332f 0,#0E1116 55%)}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:760px;padding-top:2rem}
[data-testid="stChatMessage"]{background:#171C24;border:1px solid #232B36;border-radius:22px;padding:14px 18px;margin-bottom:10px}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:#0F766E;border-color:#14998E;margin-left:12%}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) *{color:#F0FDFA}
.stButton>button{border-radius:999px;border:1px solid #2A3441;background:#171C24;color:#E6EDF3;padding:.35rem 1rem}
.stButton>button:hover{border-color:#2DD4BF;color:#2DD4BF}
[data-testid="stChatInput"]{border-radius:999px}
h1{font-weight:700;letter-spacing:-.02em}
</style>
"""


def header():
    col1, col2 = st.columns([4, 1], vertical_alignment="center")
    col1.markdown(f"## 💬 {APP_NAME}")
    st.session_state["w_lang"] = st.session_state.lang
    col2.segmented_control("lang", ["ru", "en"], key="w_lang", on_change=on_lang_change,
                           format_func=str.upper, label_visibility="collapsed")


def sidebar():
    ss = st.session_state
    base, key, model, _ = current_api()
    with st.sidebar:
        st.header(t("model_key"))
        api_settings()
        if ss.profile:
            st.divider()
            st.subheader(t("profile"))
            labels = LABELS[ss.lang]
            for k, v in ss.profile.items():
                if v:
                    st.markdown(f"**{labels[k]}:** {v}")
        st.divider()
        if st.button(t("clear"), use_container_width=True):
            ss.chat = ss.chat[:1]
            ss.pending = False
            ss.error = None
            st.rerun()
        st.button(t("restart"), use_container_width=True, on_click=reset_all)
        st.caption(t("history_note"))


def screen_chat():
    ss = st.session_state
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar()
    header()
    base, key, model, temp = current_api()
    if not (base and model and key):
        st.info(t("need_key_side"))

    questions = QUESTIONS[ss.lang]
    for role, text in ss.survey_log:
        with st.chat_message(role):
            st.write(text)

    if ss.q < len(questions):  # быстрый опрос прямо в чате
        st.progress(ss.q / len(questions), text=t("q_of", i=ss.q + 1, n=len(questions)))
        _, question, options = questions[ss.q]
        with st.chat_message("assistant"):
            st.write(question)
            if options:
                cols = st.columns(2)
                for i, opt in enumerate(options):
                    cols[i % 2].button(opt, key=f"opt_{ss.q}_{i}", on_click=answer_survey, args=(opt,),
                                       use_container_width=True)
            st.button(t("skip"), key=f"skip_{ss.q}", type="tertiary", on_click=answer_survey, args=("",))
        text = st.chat_input(t("answer_ph"))
        if text:
            answer_survey(text.strip())
            st.rerun()
        return

    mode_id = st.segmented_control(t("mode_label"), list(MODES), default="explain", key="mode",
                                   format_func=mode_label, label_visibility="collapsed") or "explain"

    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if ss.pending:
        history = [{"role": "system", "content": system_prompt(mode_id)}] + [
            {"role": m["role"], "content": m["content"]} for m in ss.chat[-CONTEXT_MESSAGES:]
        ]
        ss.pending = False
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(stream_chat(base, key, model, history, temp))
                ss.chat.append({"role": "assistant", "content": reply})
                st.rerun()
            except LLMError as e:
                ss.error = str(e)

    if ss.error:
        st.error(ss.error)
        if st.button(t("retry")):
            ss.pending = True
            ss.error = None
            st.rerun()

    if ss.chat and ss.chat[-1]["role"] == "assistant" and len(ss.chat) > 1:
        followups = FOLLOWUPS[ss.lang]
        cols = st.columns(len(followups))
        for col, (label, prompt) in zip(cols, followups):
            col.button(label, key=f"fu_{len(ss.chat)}_{label}", on_click=queue_user_message,
                       args=(prompt,), use_container_width=True)

    text = st.chat_input(t("chat_ph", mode=mode_label(mode_id)))
    if text:
        queue_user_message(text)
        st.rerun()


init_state()
screen_chat()
save_state()
