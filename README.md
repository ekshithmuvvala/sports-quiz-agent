# AI-Powered Sports Quiz Generation Agent

A Streamlit web app that generates factually grounded, multiple-choice sports
quizzes using Retrieval-Augmented Generation (RAG): a local ChromaDB vector
store of historic facts, combined with live DuckDuckGo web search for recent
events, both fed into an LLM that is instructed to answer only from that
retrieved context.

## How it works

1. You pick a sport and a difficulty level in the sidebar.
2. The agent queries **ChromaDB** for relevant historic facts about that sport.
3. The agent searches **DuckDuckGo** for recent news/results for that sport.
   Anything found is written back into ChromaDB so the knowledge base grows
   over time instead of re-searching the same facts on every request.
4. Both sources are merged into a single context block and sent to the LLM
   with strict instructions to generate questions only from that context.
5. The LLM returns structured JSON (question, four options, correct answer,
   explanation), which the app parses directly -- no fragile regex parsing.
6. Each question also gets a lightweight **grounding check**: a word-overlap
   score between its explanation and the retrieved context, so questions
   that seem to have drifted from the source material are flagged in the UI.
7. The quiz renders as an interactive card per question. Session state
   remembers previously shown questions per sport so hitting "Generate fresh
   quiz" again asks for genuinely new questions instead of repeats.

## Project structure

```
sports-quiz-agent/
├── .env.example          # Template -- copy to .env and add your real key
├── requirements.txt
├── README.md
├── data/
│   └── sports_facts.json # Local seed knowledge base (20 facts, 5 sports)
├── chroma_db/             # Created automatically by ChromaDB on first run
├── src/
│   ├── config.py          # Env vars, model name, and app-wide constants
│   ├── database.py        # All ChromaDB reads/writes -- nothing else touches it
│   ├── search.py           # All DuckDuckGo search calls
│   ├── generator.py        # RAG orchestration: retrieval + prompt + LLM call
│   └── validation.py        # Post-generation grounding check
└── app.py                 # Streamlit UI -- coordinates everything above
```

## Setup

**Prerequisites:** Python 3.9, 3.10, or 3.11 (avoid 3.12+ -- some ChromaDB
dependencies compile most reliably on these versions).

1. Create and activate a virtual environment:

   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Set up your API key:

   ```bash
   cp .env.example .env
   # then edit .env and paste your real OPENAI_API_KEY
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

   The first run will populate ChromaDB from `data/sports_facts.json`
   (creates a local `chroma_db/` folder) -- subsequent runs reuse it.

## Troubleshooting

- **ChromaDB / sqlite errors**: some environments ship an outdated system
  `sqlite3`. If you hit a version error, run
  `pip install pysqlite3-binary` -- `src/database.py` already includes a
  guarded import that swaps it in automatically if it's installed.
- **JSON parsing errors from the model**: the app uses OpenAI's JSON mode
  (`response_format={"type": "json_object"}`) so this should be rare. If it
  happens, just click "Generate fresh quiz" again.
- **API key exposure**: never hardcode your key in source. `.env` is already
  listed in `.gitignore` -- double check before pushing to a public repo.
- **Empty ChromaDB results**: if a sport's questions lean heavily on web
  search, add more entries for that sport to `data/sports_facts.json` and
  delete the `chroma_db/` folder to force a re-seed.

## Extending this project

- Swap `duckduckgo-search` for a paid provider (Tavily, SerpAPI) for more
  reliable search results at scale.
- Swap the OpenAI client in `src/generator.py` for Google's `google-genai`
  SDK to use Gemini instead.
- Replace the word-overlap grounding heuristic in `src/validation.py` with a
  second LLM call that explicitly fact-checks each answer against the
  context, if you want stronger (but slower/costlier) verification.
