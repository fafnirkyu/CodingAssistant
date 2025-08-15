## Features
- **Runs locally** with [Ollama](https://ollama.com/) — no API keys required.
- **Uses Qwen3-Coder:30B** for high-quality coding help.
- **Multi-file awareness** — includes your existing code files in context for smarter edits.
- **Web search integration** — fetches up-to-date info to help with coding questions.
- **File uploads** — drop files into the assistant to include them in the conversation context.
- **Streaming responses** — see output tokens appear in real time.
- **Project management** — work on multiple coding projects with separate histories.
- **Web UI** — chat, manage files, and code in the browser.

## Installation

### 1. Install Ollama
Download and install [Ollama](https://ollama.com/download) for your OS.

### 2. Pull Qwen3-Coder:30B model

ollama pull qwen3-coder:30b

### 3. Clone this repository

git clone https://github.com/yourusername/qwen3-coder-assistant.git
cd qwen3-coder-assistant

### 4. Install Python dependencies

pip install -r requirements.txt

### 5. Run the backend

cd backend
python app.py

### 6. Open the web UI

http://localhost:5000

Usage

Select a project from the dropdown or create a new one.

Ask coding questions — e.g. “Generate a FastAPI server with JWT authentication.”

Upload files to include them in the assistant’s context.

Enable web search to fetch live information alongside model knowledge.

Tech Stack

Backend: Python + Flask

Frontend: HTML + JavaScript

Model: Qwen3-Coder:30B via Ollama

Database: SQLite

Search API: DuckDuckGo Instant Answer API

Privacy

All prompts, files, and conversation history stay local — no data leaves your machine unless you enable web search.
