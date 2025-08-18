# Qwen Coder — Local Personal Coding Assistant

A fully local, project-aware coding companion powered by [Qwen3-Coder-30B-A3B-Instruct-480B-Distill-V2-Q5_K_M](https://huggingface.co/Qwen) running via [Ollama](https://ollama.com/).  
This assistant helps you write, debug, and manage code across multiple projects with file context awareness, streaming responses, and automatic file saving.

---

## ✨ Features

- **Local LLM** — Uses Qwen v3 30B for high-quality coding assistance, no cloud API required.
- **Project-based Context** — Automatically reads your project's files to provide relevant context in responses.
- **Multi-file Editing** — Generates and saves multiple files directly to your project folder.
- **Streaming Output** — Real-time, token-by-token response streaming.
- **Web Search** — Integrated DuckDuckGo API for quick information lookup.
- **File Uploads** — Upload project files directly via the web UI.
- **Persistent Chat Memory** — Remembers previous interactions per project using SQLite.
- **Dark Mode UI** — Clean, responsive interface with Markdown and syntax highlighting.

---

## 📂 Project Structure

.
├── backend/
│ └── app.py # Flask backend with Ollama integration
├── static/
│ ├── styles.css # Dark mode styling
│ └── script.js # Frontend interactivity & streaming
├── templates/
│ └── index.html # Web interface
├── projects/ # Your saved coding projects
├── memory.db # SQLite chat history database
└── README.md

---

## 🛠️ Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- Download the "Qwen3-Coder-30B-A3B-Instruct-480B-Distill-V2-Q5_K_M" model from Huggingface(https://huggingface.co/BasedBase/Qwen3-Coder-30B-A3B-Instruct-480B-Distill-V2)
- Put the downloaded file in the same folder as the Modelfile (/model)
- Open the CMD and run "ollama create Qwen3-Coder-30B-A3B-Instruct-480B-Distill-V2-Q5_K_M --file Modelfile"

  
Node & npm (optional, if you plan to extend the frontend)

Python dependencies:

pip install flask requests ollama

🚀 Usage
Start Ollama:

ollama serve

Run the Backend:

python backend/app.py
Open the Web UI:
Go to http://localhost:5000 in your browser.

Create a Project:
Click Add Project, then start chatting. The assistant will:

Read small project files for context

Provide explanations & runnable code

Save generated files directly into your project folder

💡 Tips

Use requirements.txt or README.md in your project — the assistant prioritizes them for context.

Press Ctrl+Enter to send messages quickly.

Uploaded files are automatically added to the active project's context.

🧩 Technology Stack

Backend

Python 3.9+ — Core backend language.

Flask — Lightweight web framework for handling API routes and serving the web UI.

SQLite — Embedded database for persistent per-project chat history.

Ollama — Local LLM runner for Qwen v3 30B.

Qwen v3 30B — Large Language Model specialized for coding and reasoning.

Requests — HTTP client for DuckDuckGo search API.

Frontend

HTML5 — Web interface structure.

CSS3 — Custom dark mode styling.

JavaScript (ES6+) — Client-side logic for chat interaction & streaming.

Fetch API — Asynchronous communication with the backend.

Marked.js — Client-side Markdown rendering for assistant messages.

Highlight.js — Syntax highlighting for code blocks.

🔒 Privacy

Everything runs entirely locally.
No data is sent to external services except optional DuckDuckGo search queries.
