# 🏦 Multi-Agent Banking Support Chatbot

**Youssef Ahmed Mohamed Shawky · 24P0132 · Generative AI**

A support chatbot for a fictional bank (NileBank). A BiLSTM intent classifier trained from scratch on Banking77
routes each message to one of three LangChain tool-calling agents. The Streamlit sidebar shows the predicted
intent, confidence, agent, LLM and tool calls live.

| Intent | Agent tool | Notes |
|---|---|---|
| **billing** | `check_invoice` (mock account DB) | charges, fees, refunds, transfers |
| **technical** | `search_docs` (RAG) | card not working, PIN, passcode, security |
| **general** | `search_docs` (RAG) | delivery, countries, age limit + fallback when confidence < 60% |

**Results (Banking77 test set, 3,080 messages):** accuracy **92.0%**, macro-F1 **0.906**.

## LLM providers
- **Primary:** Groq, `openai/gpt-oss-120b` (the brief's Llama models were retired by Groq in August 2026)
- **Fallback:** Ollama (local), `llama3.2:3b`

## Setup
```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```
Copy `.env.example` to `.env` and add your free key from [console.groq.com](https://console.groq.com):
```
GROQ_API_KEY=your_key_here
SIMULATE_RATE_LIMIT=0
```
Optional fallback: install [Ollama](https://ollama.com), then run `ollama pull llama3.2:3b`.

## Run (from the project root)
| Task | Command |
|---|---|
| Start the app | `python -m streamlit run app.py` |
| Test the router in the terminal | `python -m agents.router` |
| Train the classifier | `python training/train.py` |
| Evaluate | `python training/evaluate.py` |

**Try:** `I was charged twice for the same payment, my customer id is C001` · `My card is not working` ·
`How long does card delivery take?` · `hi`

**Rate-limit demo:** set `SIMULATE_RATE_LIMIT=1` in `.env` and restart; replies then come from Ollama.

## Structure
```
app.py       Streamlit UI
agents/      router, agents (prompts), tools, memory, llm
training/    data prep, preprocessing, model, train, evaluate, classifier.pt
data/        train/val/test CSVs + labeled.md (77 → 3 label map)
kb/          15 Markdown documents for RAG
```

## Notes
- `.env` is not included; use `.env.example`.
- AI-assisted code sections are marked in the source files with `AI GENERATED` comments.
