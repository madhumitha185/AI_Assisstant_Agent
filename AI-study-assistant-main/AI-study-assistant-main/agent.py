"""
Agent assembly: LLM + tools + conversation memory using LangChain's tool-calling agent.
"""
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from tools import make_search_tool, make_quiz_tool, make_schedule_tool

SYSTEM_PROMPT = """You are StudyMate, an AI Learning & Study Assistant built for an IBM internship demo.

You have three tools:
1. search_study_materials - retrieves relevant chunks from the student's uploaded notes/syllabus.
2. generate_quiz - writes a 3-question MCQ quiz on a topic, grounded in retrieved material.
3. calculate_study_schedule - builds a day-by-day revision timetable.

Rules:
- For ANY question about course content, ALWAYS call search_study_materials first, then answer
  using only that retrieved context. Cite sources inline as [Source N: filename].
- If the retrieved material doesn't cover the question, say so honestly instead of guessing.
- If the student asks to be quizzed or tested, call generate_quiz.
- If the student asks for a study plan, schedule, or timetable, call calculate_study_schedule with
  the format 'topics=...|...; days=N; hours_per_day=N', inferring topics from context if not given.
- Keep answers concise, encouraging, and student-friendly.
- Use conversation history to resolve follow-ups like "explain that more" or "quiz me on that".
"""


def build_llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file (see .env.example) "
            "and restart the app."
        )
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.3,
    )


def build_agent_executor(retriever) -> AgentExecutor:
    """Assemble the full agent: LLM + tools + memory + prompt."""
    llm = build_llm()
    tools = [
        make_search_tool(retriever),
        make_quiz_tool(retriever, llm),
        make_schedule_tool(),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="output"
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )
    return executor
