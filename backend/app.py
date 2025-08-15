# backend/app.py
import os
import re
import glob
import time
import sqlite3
from flask import Flask, request, jsonify, render_template, Response, abort
import ollama
import requests
from werkzeug.utils import secure_filename


# ---------------- Config ----------------
MODEL = os.getenv("MODEL", "qwen3-coder:30b")
DB_PATH = os.getenv("DB_PATH", "memory.db")
PROJECTS_DIR = os.getenv("PROJECTS_DIR", "./projects")

# Max files/size to include in prompt context
MAX_FILES_IN_CONTEXT = int(os.getenv("MAX_FILES_IN_CONTEXT", "40"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(8 * 1024)))  # 8KB each

SYSTEM_PROMPT = """
You are a senior software engineer AI.

Goals:
- Write complete, working, tested code.
- When given a task, think through steps internally before answering.
- Include explanations unless the user says “just code”.
- Use full, runnable examples with all necessary imports.
- Prefer clear, maintainable, idiomatic code and best practices.

Multi-file output format:
For each file, output exactly:

FILE: relative/path/to/file.ext
```language
<full file contents>
```

Rules:
- Use Markdown code fences with language labels (e.g., ```python).
- Do not omit imports or helper functions.
- If modifying existing files, output the full revised file.
- Before finalizing, review your code for correctness and missing pieces.
"""

# -------------- Flask app ---------------
app = Flask(__name__, static_folder="../static", template_folder="../templates")

# -------------- SQLite memory -----------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT,
    role TEXT,
    content TEXT,
    ts REAL
)
""")
conn.commit()

def save_message(project: str, role: str, content: str):
    """Insert a message into the DB and ensure the project is registered."""
    cur.execute(
        "INSERT INTO messages (project, role, content, ts) VALUES (?, ?, ?, ?)",
        (project, role, content, time.time())
    )
    conn.commit()

def load_recent(project: str, limit: int = 12):
    cur.execute(
        "SELECT role, content FROM messages WHERE project=? ORDER BY id DESC LIMIT ?",
        (project, limit)
    )
    rows = cur.fetchall()[::-1]  # chronological order
    return [{"role": r[0], "content": r[1]} for r in rows]

# -------------- Project FS helpers --------------
def project_base_dir(project: str) -> str:
    base = os.path.abspath(PROJECTS_DIR)
    path = os.path.abspath(os.path.join(base, project))
    # Prevent path traversal outside the projects dir
    if not path.startswith(base + os.sep) and path != base:
        abort(400, description="Invalid project path")
    os.makedirs(path, exist_ok=True)
    return path

def load_project_files_context(project: str) -> str:
    """Read a subset of small project files and format them for prompt context."""
    base_dir = project_base_dir(project)
    files_data = []
    count = 0

    preferred_globs = [
        "**/*.py", "**/*.ipynb", "**/*.md",
        "**/*.js", "**/*.ts", "**/*.tsx", "**/*.jsx",
        "**/*.json", "**/*.yml", "**/*.yaml",
        "**/*.toml", "**/*.ini",
        "**/*.html", "**/*.css",
        "README.md", "requirements.txt", "pyproject.toml", "setup.py"
    ]

    seen = set()
    for pattern in preferred_globs:
        for path in glob.glob(os.path.join(base_dir, pattern), recursive=True):
            if count >= MAX_FILES_IN_CONTEXT:
                break
            if not os.path.isfile(path):
                continue
            if path in seen:
                continue
            seen.add(path)

            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > MAX_FILE_BYTES:
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            rel_path = os.path.relpath(path, base_dir)
            files_data.append(f"FILE: {rel_path}\n```\n{content}\n```")
            count += 1

        if count >= MAX_FILES_IN_CONTEXT:
            break

    if not files_data:
        return "No existing project files found."
    return "\n\n".join(files_data)

FILE_BLOCK_RE = re.compile(
    r"FILE:\s*(?P<path>[^\n\r]+)\s*```(?:[\w.+-]+)?\s*\n(?P<code>.*?)```",
    re.DOTALL
)

def save_generated_files(project: str, assistant_text: str):
    base_dir = project_base_dir(project)
    matches = FILE_BLOCK_RE.findall(assistant_text)
    for rel_path, code in matches:
        rel_path = rel_path.strip().replace("\\", "/")
        abs_path = os.path.abspath(os.path.join(base_dir, rel_path))
        if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
            continue
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(code.strip())            

def build_full_prompt(project: str, user_text: str) -> str:
    hist = load_recent(project, limit=8)
    hist_lines = []
    for m in hist:
        hist_lines.append(f"{m['role'].upper()}: {m['content']}")

    files_context = load_project_files_context(project)

    prompt_sections = [
        SYSTEM_PROMPT.strip(),
        "\n--- Project Context ---\n",
        files_context,
        "\n--- Conversation (recent) ---\n",
        "\n".join(hist_lines),
        "\n--- New Request ---\n",
        f"USER: {user_text}\nASSISTANT:"
    ]
    return "\n".join(s for s in prompt_sections if s is not None)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history/<project>")
def history(project):
    return jsonify(load_recent(project, limit=200))

@app.route("/projects")
def get_projects():
    cur.execute("SELECT DISTINCT project FROM messages ORDER BY project ASC")
    rows = cur.fetchall()
    return jsonify([r[0] for r in rows])

@app.route("/add_project", methods=["POST"])
def add_project():
    data = request.json or {}
    project = (data.get("project") or "").strip()
    if not project:
        return jsonify({"error": "empty project name"}), 400
    project_base_dir(project)
    save_message(project, "system", f"Project {project} created.")
    return jsonify({"status": "ok", "project": project})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    project = data.get("project", "default")
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"error": "empty message"}), 400

    save_message(project, "user", user_text)
    full_prompt = build_full_prompt(project, user_text)

    try:
        resp = ollama.generate(model=MODEL, prompt=full_prompt, stream=False)
        assistant_text = resp.get("response") or resp.get("text") or ""
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    save_message(project, "assistant", assistant_text)
    save_generated_files(project, assistant_text)
    return jsonify({"response": assistant_text})

@app.route("/stream", methods=["POST"])
def stream():
    data = request.json or {}
    project = data.get("project", "default")
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return Response("data: ERROR: empty message\n\n", mimetype="text/event-stream")

    save_message(project, "user", user_text)
    full_prompt = build_full_prompt(project, user_text)

    def generate():
        try:
            stream = ollama.generate(model=MODEL, prompt=full_prompt, stream=True)
            assistant_text = ""
            for chunk in stream:
                token = chunk.get("response") or ""
                if token:
                    assistant_text += token
                    safe_token = token.replace("\r", "").replace("\n", "\\n")
                    yield f"data: {safe_token}\n\n"

            save_message(project, "assistant", assistant_text)
            save_generated_files(project, assistant_text)

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/search_web", methods=["POST"])
def search_web():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400

    r = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
        timeout=5
    )
    try:
        ddg = r.json()
    except Exception:
        return jsonify({"error": "failed to parse search results"}), 500

    results = []
    if ddg.get("AbstractText"):
        results.append(ddg["AbstractText"])
    for t in ddg.get("RelatedTopics", []):
        if isinstance(t, dict) and "Text" in t:
            results.append(t["Text"])

    return jsonify({"results": results})

ALLOWED_EXTENSIONS = {"py", "js", "ts", "tsx", "jsx", "md", "txt", "json", "yml", "yaml", "html", "css"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_file/<project>", methods=["POST"])
def upload_file(project):
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(project_base_dir(project), filename)
        file.save(save_path)
        save_message(project, "system", f"File uploaded: {filename}")
        return jsonify({"status": "ok", "filename": filename})
    else:
        return jsonify({"error": "file type not allowed"}), 400

if __name__ == "__main__":
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
