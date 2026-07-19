"""
The RAG orchestration layer. This is the only module that talks to the LLM.
It pulls context from the two retrieval sources, builds the grounded prompt,
calls the model, parses the structured output, and runs the grounding check
-- app.py never touches ChromaDB, DuckDuckGo, or the OpenAI client directly.
"""

import json

from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL, NUM_QUESTIONS
from src.database import query_historic_facts, add_fresh_fact
from src.search import get_live_news_context
from src.validation import check_grounding


def _build_system_instruction(unified_context):
    return (
        "You are an expert sports quiz creator. Write multiple-choice quiz "
        "questions relying strictly on the provided context. Do not use facts "
        "that are not present in the context below. If the context is thin, "
        "write fewer confident questions rather than inventing details.\n\n"
        "Respond with ONLY a JSON object (no markdown fences, no commentary) "
        "matching exactly this shape:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question": "string",\n'
        '      "options": {"A": "string", "B": "string", "C": "string", "D": "string"},\n'
        '      "correct_answer": "A" | "B" | "C" | "D",\n'
        '      "explanation": "string, should reference the supporting fact"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"CONTEXT:\n{unified_context}"
    )


def _build_user_prompt(sport, difficulty, previous_questions):
    prompt = (
        f"Generate exactly {NUM_QUESTIONS} unique multiple-choice questions "
        f"for the sport: {sport}.\nDifficulty target: {difficulty}."
    )
    if previous_questions:
        joined = "; ".join(previous_questions[-10:])
        prompt += (
            "\n\nDo not repeat or closely rephrase any of these questions "
            f"already shown to the user: {joined}"
        )
    return prompt


def compile_quiz_data(sport, difficulty, previous_questions=None):
    """
    1. Retrieves historical facts from ChromaDB.
    2. Retrieves live context from DuckDuckGo.
    3. Writes any new web facts back into ChromaDB so the store grows.
    4. Builds a grounded prompt and calls the LLM for structured JSON output.
    5. Runs a grounding check on the result.

    Returns (questions, unified_context) where questions is a list of dicts
    already annotated with grounding_score / grounded.
    """
    db_query = f"{sport} history cup championships rules records"
    db_matches = query_historic_facts(sport=sport, query_text=db_query)
    db_context = "\n".join(db_matches) if db_matches else "No offline historic data recorded."

    web_context, web_snippets = get_live_news_context(sport)

    # Feed fresh web findings back into the vector store for next time.
    for snippet in web_snippets:
        add_fresh_fact(sport, snippet)

    unified_context = (
        f"=== HISTORICAL FACTS ===\n{db_context}\n\n"
        f"=== LIVE INTERNET NEWS ===\n{web_context}"
    )

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _build_system_instruction(unified_context)},
            {"role": "user", "content": _build_user_prompt(sport, difficulty, previous_questions)},
        ],
        temperature=0.7,
    )

    raw_content = response.choices[0].message.content

    try:
        parsed = json.loads(raw_content)
        questions = parsed.get("questions", [])
    except json.JSONDecodeError:
        # Even with JSON mode this is defensive -- surface a clear error to
        # the UI instead of a raw stack trace.
        raise ValueError(
            "The model did not return valid JSON. Try regenerating the quiz."
        )

    questions = check_grounding(questions, unified_context)
    return questions, unified_context
