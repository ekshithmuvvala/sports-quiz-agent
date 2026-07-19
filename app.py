import streamlit as st

from src.config import SUPPORTED_SPORTS, DIFFICULTY_LEVELS
from src.database import setup_and_populate_db
from src.generator import compile_quiz_data


@st.cache_resource
def prepare_knowledge_base():
    """Runs once per server process -- seeds ChromaDB from the local JSON
    facts file if it hasn't been populated yet."""
    setup_and_populate_db()


prepare_knowledge_base()

st.set_page_config(page_title="Sports Quiz Agent", page_icon=":trophy:", layout="centered")

st.title("AI-Powered Sports Quiz Generator")
st.write(
    "Challenge yourself or generate engaging social media content, "
    "powered by RAG (ChromaDB + live web search)."
)

# --- Sidebar controls -----------------------------------------------------
st.sidebar.header("Quiz settings")
sport_choice = st.sidebar.selectbox("Select sport", SUPPORTED_SPORTS)
difficulty = st.sidebar.select_slider("Select difficulty", options=DIFFICULTY_LEVELS)

# --- Session state ----------------------------------------------------------
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
    st.session_state.quiz_context = None
    st.session_state.quiz_sport = None
    st.session_state.quiz_difficulty = None
    # Tracks question text already shown per sport, so "regenerate" doesn't
    # just re-serve the same four questions.
    st.session_state.seen_questions = {}

generate_clicked = st.sidebar.button("Generate fresh quiz", use_container_width=True)

if generate_clicked:
    with st.spinner("Fetching historical facts and scouring the live web..."):
        try:
            previous = st.session_state.seen_questions.get(sport_choice, [])
            questions, context_used = compile_quiz_data(
                sport=sport_choice,
                difficulty=difficulty,
                previous_questions=previous,
            )

            st.session_state.quiz_questions = questions
            st.session_state.quiz_context = context_used
            st.session_state.quiz_sport = sport_choice
            st.session_state.quiz_difficulty = difficulty

            seen = st.session_state.seen_questions.setdefault(sport_choice, [])
            seen.extend(q["question"] for q in questions if "question" in q)

            st.success("Quiz created successfully.")
        except Exception as e:
            st.error(f"Failed to generate quiz: {e}")

# --- Display the quiz -------------------------------------------------------
if st.session_state.quiz_questions:
    st.subheader(f"Current quiz: {st.session_state.quiz_sport} ({st.session_state.quiz_difficulty})")

    for i, q in enumerate(st.session_state.quiz_questions, start=1):
        st.markdown(f"**Q{i}. {q.get('question', '')}**")

        options = q.get("options", {})
        labels = [f"{key}) {value}" for key, value in options.items()]

        choice = st.radio(
            "Choose an answer:",
            labels,
            key=f"q_{i}_{st.session_state.quiz_sport}_{st.session_state.quiz_difficulty}",
            index=None,
        )

        if choice is not None:
            chosen_key = choice.split(")")[0]
            correct_key = q.get("correct_answer", "")

            if chosen_key == correct_key:
                st.success(f"Correct! {q.get('explanation', '')}")
            else:
                st.error(f"Not quite -- the correct answer is {correct_key}. {q.get('explanation', '')}")

        if q.get("grounded"):
            st.caption(f"Grounded in retrieved context (score {q.get('grounding_score')})")
        else:
            st.caption(f"⚠ Low grounding confidence (score {q.get('grounding_score')}) -- verify before posting")

        st.divider()

    with st.expander("Inspect ground truth (RAG context used)"):
        st.code(st.session_state.quiz_context, language="markdown")
else:
    st.info("Pick a sport and difficulty in the sidebar, then click 'Generate fresh quiz' to begin.")
