"""
Custom agent tools:
 1. search_study_materials  -> RAG retrieval over the vector store
 2. generate_quiz           -> LLM-generated 3-question MCQ quiz grounded in retrieved context
 3. calculate_study_schedule-> deterministic (non-LLM) day-by-day schedule builder
"""
from typing import List
from langchain_core.tools import Tool


def make_search_tool(retriever) -> Tool:
    """Build a tool that lets the agent search the uploaded study materials."""

    def _search(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant material found in the knowledge base. Ask the user to upload notes."
        blocks = []
        for i, d in enumerate(docs, start=1):
            source = d.metadata.get("source", "unknown")
            blocks.append(f"[Source {i}: {source}]\n{d.page_content.strip()}")
        return "\n\n---\n\n".join(blocks)

    return Tool(
        name="search_study_materials",
        func=_search,
        description=(
            "Search the student's uploaded syllabus and notes for information relevant "
            "to a question or topic. ALWAYS call this before answering any academic "
            "question, and cite the returned [Source N: filename] tags in your final answer."
        ),
    )


def make_quiz_tool(retriever, llm) -> Tool:
    """Build a tool that generates a 3-question MCQ quiz from retrieved context."""

    def _generate_quiz(topic: str) -> str:
        docs = retriever.invoke(topic)
        if not docs:
            return "No study material found on that topic. Ask the user to upload relevant notes first."
        context = "\n\n".join(d.page_content for d in docs[:4])
        prompt = f"""You are a strict exam question writer. Using ONLY the study material
below, write exactly 3 multiple-choice questions (MCQs) about "{topic}".

Rules:
- Each question must have 4 options labeled A, B, C, D.
- Mark the correct answer clearly at the end of each question as "Answer: X".
- Base every question strictly on the material given; do not invent facts not present in it.
- Number the questions 1, 2, 3.

STUDY MATERIAL:
{context}

Return only the quiz text, nothing else."""
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        return f"QUIZ_START\n{content.strip()}\nQUIZ_END"

    return Tool(
        name="generate_quiz",
        func=_generate_quiz,
        description=(
            "Generate a 3-question multiple-choice quiz on a given topic, grounded in the "
            "student's uploaded study material. Input should be the topic or subject name. "
            "Use this whenever the student asks to be quizzed, tested, or wants practice questions."
        ),
    )


def make_schedule_tool() -> Tool:
    """Build a deterministic (non-LLM) study schedule calculator tool."""

    def _calculate_schedule(plan_input: str) -> str:
        """
        Expected plan_input format (semicolon-separated key=value pairs):
        "topics=Topic1|Topic2|Topic3; days=5; hours_per_day=2"
        Falls back to treating the whole string as a comma-separated topic list.
        """
        topics: List[str] = []
        days = 5
        hours_per_day = 2.0

        try:
            parts = [p.strip() for p in plan_input.split(";") if p.strip()]
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key == "topics":
                    topics = [t.strip() for t in value.split("|") if t.strip()]
                elif key == "days":
                    days = max(1, int(float(value)))
                elif key in ("hours_per_day", "hours"):
                    hours_per_day = max(0.5, float(value))
        except Exception:
            pass

        if not topics:
            topics = [
                t.strip()
                for t in plan_input.replace(";", ",").split(",")
                if t.strip() and "=" not in t
            ]
        if not topics:
            topics = ["General Review"]

        schedule = {d: [] for d in range(1, days + 1)}
        for i, topic in enumerate(topics):
            day = (i % days) + 1
            schedule[day].append(topic)

        lines = [f"### {days}-Day Study Schedule ({hours_per_day} hrs/day)\n"]
        lines.append("| Day | Topics | Est. Hours |")
        lines.append("|-----|--------|------------|")
        for day in range(1, days + 1):
            day_topics = schedule[day] if schedule[day] else ["Buffer / Revision"]
            topics_str = ", ".join(day_topics)
            lines.append(f"| Day {day} | {topics_str} | {round(hours_per_day, 1)}h |")

        total_hours = round(days * hours_per_day, 1)
        lines.append(f"\n**Total study time:** {total_hours} hours across {days} days.")
        lines.append("Tip: briefly revisit Day 1 topics on the final day (spaced repetition).")
        return "\n".join(lines)

    return Tool(
        name="calculate_study_schedule",
        func=_calculate_schedule,
        description=(
            "Calculate a day-by-day study schedule. Input must be a single string in the "
            "exact format: 'topics=Topic1|Topic2|Topic3; days=5; hours_per_day=2'. "
            "Use this whenever the student asks for a study plan, revision timetable, or "
            "how to divide their time before an exam."
        ),
    )
