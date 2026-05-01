from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os, json
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def evaluate_answer(question: str, answer: str, resume_text: str) -> dict:
    prompt = f"""
You are an expert interviewer evaluating the candidate's resume.

Resume context:
{resume_text[:1000]}

Question asked: {question}
Candidate answer: {answer}

Return ONLY a valid JSON object, no extra text, no markdown:
{{
  "score": <integer 1-10>,
  "feedback": "<2 sentence honest feedback>",
  "weak_area": "<specific topic they are weak in, or null>",
  "strong_point": "<what they did well>"
}}
"""
    response = llm.invoke([
        SystemMessage(content="You are a strict but fair technical interviewer. Return only JSON."),
        HumanMessage(content=prompt)
    ])
    try:
        text = response.content.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except:
        return {
            "score": 5,
            "feedback": response.content[:200],
            "weak_area": None,
            "strong_point": "Attempted the question"
        }


def generate_gap_analysis(scores: list, weak_areas: list, resume_text: str) -> str:
    avg = sum(scores) / len(scores) if scores else 0
    areas = [a for a in weak_areas if a]
    prompt = f"""
You are an expert career coach reviewing a mock interview session.

Candidate resume:
{resume_text[:800]}

Interview results:
- Average score: {avg:.1f}/10
- Weak areas identified: {', '.join(areas) if areas else 'None detected'}
- Questions answered: {len(scores)}
- All scores: {scores}

Write a Gap Analysis report with these sections:
1. Overall Performance Summary (2-3 sentences)
2. Strong Areas (bullet points)
3. Weak Areas and How to Improve (with specific resources relevant to THEIR field)
4. Top 3 Action Items for next 2 weeks
5. Job Readiness Score: X/10

IMPORTANT: Base all feedback strictly on the candidate's actual field and resume.
Do NOT suggest AI/ML certifications or developer skills unless they are already 
in the candidate's resume. Tailor everything to their actual profession.
"""
    response = llm.invoke([
        SystemMessage(content="You are a career coach who understand the candidate's resume and ask questions to test them."),
        HumanMessage(content=prompt)
    ])
    return response.content
