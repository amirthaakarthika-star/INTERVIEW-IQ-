# InterviewIQ - AI Mock Interview Coach

InterviewIQ is an AI-powered mock interview coach that generates personalized interview questions from a candidate's resume. It parses an uploaded PDF resume, asks technical and behavioral questions, evaluates answers, gives scores, and creates a final gap analysis report.

## Features

- Upload a resume as a PDF
- Generate interview questions based on the resume
- Run a 6-question mock interview flow
- Score each answer out of 10
- Give strengths and improvement areas after every response
- Generate a final interview gap analysis
- Simple FastAPI backend with a static HTML frontend

## Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Backend programming language |
| FastAPI | API server for resume upload, chat, and reset endpoints |
| LangGraph | Interview flow and state-based workflow logic |
| LangChain | LLM message handling and integration utilities |
| Groq LLM API | AI model provider for questions, evaluation, and gap analysis |
| pypdf | Extracts text from uploaded PDF resumes |
| HTML | Frontend page structure |
| CSS | Styling and responsive layout |
| JavaScript | Browser-side chat, upload, and API interaction logic |

## Project Structure

```text
INTERVIEW-IQ-
+-- main.py              # FastAPI backend and interview flow
+-- evaluator.py         # Answer evaluation and gap analysis logic
+-- resume_parser.py     # PDF resume parser
+-- index.html           # Frontend UI
+-- requirements.txt     # Python dependencies
+-- .gitignore
+-- README.md
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/amirthaakarthika-star/INTERVIEW-IQ-.git
cd INTERVIEW-IQ-
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Do not commit your `.env` file. It is already included in `.gitignore`.

## Run the App

Start the FastAPI backend:

```bash
uvicorn main:app --reload
```

The backend will run at:

```text
http://localhost:8000
```

Then open `index.html` in your browser and upload a PDF resume to begin the mock interview.

## API Endpoints

### `GET /`

Checks whether the backend is running.

### `POST /start`

Uploads and parses a resume PDF, then starts a new interview session.

### `POST /chat`

Sends the candidate's answer, evaluates it, and returns the next question or final report.

### `POST /reset`

Resets the current interview session.

## Notes

- The current version is designed as a local prototype.
- It stores interview state in memory, so it is best for one user at a time.
- For production, add proper session handling, unique resume upload paths, stronger file validation, and restricted CORS settings.

## License

This project is for learning and portfolio use.
