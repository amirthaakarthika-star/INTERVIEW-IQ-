from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from resume_parser import parse_resume
from evaluator import evaluate_answer, generate_gap_analysis
import os, shutil
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
class InterviewState(TypedDict):
    resume_path: str
    resume_data: dict
    round: str
    question_count: int
    current_question: str
    conversation_history: List[dict]
    scores: List[int]
    weak_areas: List[str]
    strong_points: List[str]
    gap_analysis: Optional[str]
    response_to_user: str
    interview_complete: bool
    last_action: str  # "start" | "answer"

# ─────────────────────────────────────────
# NODE 1 — Resume Parser
# ─────────────────────────────────────────
def resume_parser_node(state: InterviewState) -> InterviewState:
    resume_data = parse_resume(state["resume_path"])
    state["resume_data"] = resume_data
    state["round"] = "technical"
    state["question_count"] = 0
    state["scores"] = []
    state["weak_areas"] = []
    state["strong_points"] = []
    state["conversation_history"] = []
    state["interview_complete"] = False

    # Generate personalised opening from resume
    resume_text = resume_data["raw_text"][:1500]
    response = llm.invoke([
        SystemMessage(content="You are a professional job interviewer. Adapt your interview style based on the candidate's resume and their field. Be warm but professional."),
        HumanMessage(content=f"""
The candidate has uploaded their resume. Here it is:
{resume_text}

Write a short greeting (3-4 sentences) that:
1. Welcomes them by name if you can find it
2. Mentions 1-2 specific things you noticed in their resume (projects, skills)
3. Explains the interview structure: 4 technical questions then 2 behavioural questions
4. Ends with the first technical question about their most impressive project

Keep it conversational and encouraging.
""")
    ])

    first_q = "Tell me about your most impressive project and the biggest technical challenge you faced."
    state["current_question"] = first_q
    state["response_to_user"] = response.content
    state["last_action"] = "start"
    return state

# ─────────────────────────────────────────
# NODE 2 — Evaluator (runs after every answer)
# ─────────────────────────────────────────
def evaluator_node(state: InterviewState) -> InterviewState:
    history = state["conversation_history"]
    if not history:
        return state

    last = history[-1]
    if last["role"] != "user":
        return state

    evaluation = evaluate_answer(
        state["current_question"],
        last["content"],
        state["resume_data"]["raw_text"]
    )

    state["scores"].append(evaluation["score"])
    if evaluation.get("weak_area"):
        state["weak_areas"].append(evaluation["weak_area"])
    if evaluation.get("strong_point"):
        state["strong_points"].append(evaluation["strong_point"])

    feedback = (
        f"\n\n---\n"
        f"📊 **Score: {evaluation['score']}/10**\n"
        f"{evaluation['feedback']}\n"
    )
    if evaluation.get("weak_area"):
        feedback += f"⚠️ **Improve:** {evaluation['weak_area']}\n"
    if evaluation.get("strong_point"):
        feedback += f"✅ **Strong:** {evaluation['strong_point']}\n"
    feedback += "---\n\n"

    state["question_count"] += 1
    state["last_action"] = "evaluated"

    # Store feedback to prepend to next question
    state["response_to_user"] = feedback
    return state

# ─────────────────────────────────────────
# NODE 3 — Technical Round
# ─────────────────────────────────────────
def technical_round_node(state: InterviewState) -> InterviewState:
    resume_text = state["resume_data"]["raw_text"][:1500]
    q_num = state["question_count"]

    # Generate question from the actual resume every time
    response = llm.invoke([
        SystemMessage(content="You are a professional interviewer. Ask technical questions relevant to the candidate's actual field and projects mentioned in their resume. Do not assume they are a developer unless their resume says so."),
        HumanMessage(content=f"""
Based on this resume, ask one technical question about their specific projects or skills.
Do not ask about RAG or LangChain unless they are mentioned in the resume.
Only ask about what is actually written in the resume.

Resume:
{resume_text}

This is question number {q_num + 1} of 4. Ask something different from previous questions.
Previous questions count: {q_num}
""")
    ])

    next_q = response.content.strip()
    state["current_question"] = next_q
    state["response_to_user"] += f"**Technical Question {q_num + 1}:** {next_q}"
    return state

# ─────────────────────────────────────────
# NODE 4 — Behavioural Round
# ─────────────────────────────────────────
def behavioural_round_node(state: InterviewState) -> InterviewState:
    q_num = state["question_count"]  # 4 or 5

    if q_num == 4:
        state["round"] = "behavioural"
        next_q = "Tell me about a time you faced a difficult technical challenge. Use the STAR method: Situation, Task, Action, Result."
        state["response_to_user"] += f"Great technical round! Now moving to Behavioural 🤝\n\n**Behavioural Question 1:** {next_q}"
    else:
        next_q = "Why do you want to work at this role and where do you see yourself in 2 years?"
        state["response_to_user"] += f"**Behavioural Question 2:** {next_q}"

    state["current_question"] = next_q
    return state

# ─────────────────────────────────────────
# NODE 5 — Gap Analysis
# ─────────────────────────────────────────
def gap_analysis_node(state: InterviewState) -> InterviewState:
    analysis = generate_gap_analysis(
        state["scores"],
        state["weak_areas"],
        state["resume_data"]["raw_text"]
    )
    state["gap_analysis"] = analysis
    state["interview_complete"] = True
    state["response_to_user"] = f"# 📋 Your Interview Gap Analysis\n\n{analysis}"
    return state

# ─────────────────────────────────────────
# ROUTER — decides which node runs next
# ─────────────────────────────────────────
def router(state: InterviewState) -> str:
    q = state["question_count"]
    if q >= 6:
        return "gap_analysis"
    elif q >= 4:
        return "behavioural_round"
    else:
        return "technical_round"

# ─────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────
def build_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("resume_parser", resume_parser_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("technical_round", technical_round_node)
    graph.add_node("behavioural_round", behavioural_round_node)
    graph.add_node("gap_analysis", gap_analysis_node)

    graph.set_entry_point("resume_parser")
    graph.add_edge("resume_parser", END)

    graph.add_conditional_edges(
        "evaluator",
        router,
        {
            "technical_round": "technical_round",
            "behavioural_round": "behavioural_round",
            "gap_analysis": "gap_analysis",
        }
    )

    graph.add_edge("technical_round", END)
    graph.add_edge("behavioural_round", END)
    graph.add_edge("gap_analysis", END)

    return graph.compile()

interview_graph = build_graph()

# Global state store (one session at a time)
current_state: InterviewState = {}

# ─────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────
class Message(BaseModel):
    message: str

@app.post("/start")
async def start(resume: UploadFile = File(...)):
    global current_state

    # Save uploaded PDF temporarily
    temp_path = f"temp_resume.pdf"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(resume.file, f)

    initial_state: InterviewState = {
        "resume_path": temp_path,
        "resume_data": {},
        "round": "technical",
        "question_count": 0,
        "current_question": "",
        "conversation_history": [],
        "scores": [],
        "weak_areas": [],
        "strong_points": [],
        "gap_analysis": None,
        "response_to_user": "",
        "interview_complete": False,
        "last_action": "start"
    }

    current_state = interview_graph.invoke(initial_state)

    return {
        "reply": current_state["response_to_user"],
        "round": current_state["round"],
        "question_count": current_state["question_count"],
        "scores": current_state["scores"]
    }



@app.post("/chat")
def chat(msg: Message):
    global current_state

    if current_state.get("interview_complete"):
            scores = current_state.get("scores", [])
            avg = sum(scores) / len(scores) if scores else 0

            if avg < 7:
                closing = (
                    f"Your average score was {avg:.1f}/10.\n\n"
                    "💪 You're not quite ready yet — but that's okay! "
                    "Practice more, review the weak areas in your Gap Analysis report, "
                    "and come back for another session. Consistency is the key!\n\n"
                    "Click '↺ New session' to practice again. You've got this! 🔥"
                )
            else:
                closing = (
                    f"Your average score was {avg:.1f}/10.\n\n"
                    "🎉 Great job! You're ready for real interviews! "
                    "Keep practicing to stay sharp.\n\n"
                    "Click '↺ New session' to practice with a new resume! 💪"
                )

            return {
                "reply": closing,
                "round": current_state["round"],
                "question_count": current_state["question_count"],
                "scores": current_state["scores"],
                "complete": True
            }

    current_state["conversation_history"].append({
        "role": "user",
        "content": msg.message
    })

    # Step 1: evaluate the answer
    current_state = evaluator_node(current_state)

    # Step 2: route to next question based on count
    q = current_state["question_count"]
    if q >= 6:
        current_state = gap_analysis_node(current_state)
    elif q >= 4:
        current_state = behavioural_round_node(current_state)
    else:
        current_state = technical_round_node(current_state)

    return {
        "reply": current_state["response_to_user"],
        "round": current_state["round"],
        "question_count": current_state["question_count"],
        "scores": current_state["scores"],
        "complete": current_state["interview_complete"]
    }

@app.post("/reset")
def reset():
    global current_state
    if os.path.exists("temp_resume.pdf"):
        os.remove("temp_resume.pdf")
    current_state = {}
    return {"status": "reset"}

@app.get("/")
def root():
    return {"status": "InterviewIQ backend running"}
