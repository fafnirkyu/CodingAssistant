import os
import re
import glob
import time
import json
import sqlite3
import subprocess
from typing import List, Dict, Tuple, Optional
from flask import Flask, request, jsonify, render_template, Response, abort
import ollama
import requests
from werkzeug.utils import secure_filename
# ---------------------------
# Config
# ---------------------------
MODEL = os.getenv("MODEL", "Qwen3-Coder-30B-A3B-Instruct-480B-Distill-V2-Q5_K_M")
DB_PATH = os.getenv("DB_PATH", "memory.db")
PROJECTS_DIR = os.getenv("PROJECTS_DIR", "./projects")

# LLM sampling/ctx controls
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
NUM_CTX = int(os.getenv("NUM_CTX", "32768"))        
SEED = int(os.getenv("SEED", "7"))                  

# Prompt/Context budgets
MAX_FILES_IN_CONTEXT = int(os.getenv("MAX_FILES_IN_CONTEXT", "80"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(64 * 1024)))  # 64KB each
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", str(120_000)))

# Save/write guardrails
ALLOWED_EXTENSIONS = {
    "py", "ipynb", "js", "ts", "tsx", "jsx", "md", "txt", "json", "yml", "yaml",
    "html", "css", "toml", "ini", "cfg", "sh", "ps1"
}

# Optional local runner/linters (disabled by default)
RUNNER_ENABLED = os.getenv("RUNNER_ENABLED", "0") == "1"
LINTER_ENABLED = os.getenv("LINTER_ENABLED", "0") == "1"
# ---------------------------
# Response contract/prompt
# ---------------------------
SYSTEM_PROMPT = """
You are a senior AI software engineer with expertise in:
- Machine Learning & Data Science (Python, sklearn, XGBoost, PyTorch, TensorFlow, pandas, numpy, matplotlib, etc.)
- General Software Engineering (Python, JavaScript/TypeScript, HTML/CSS, backend systems, APIs)

Non-negotiable contract:
1) Start with an explicit '#mode: write|review|explain' line chosen to match the user's ask. Default '#mode: write'.
2) Then output a short plan in 3–7 bullets under a heading 'Plan'.
3) When writing code, use the Multi-file format EXACTLY as specified below. Every file MUST be wrapped in a 'FILE:' header and a language-tagged code fence.
4) Produce complete, runnable, minimal examples (MWE) when relevant.
5) Prefer clear, maintainable, idiomatic code using PEP8, type hints, and docstrings. Avoid partial snippets.

## Multi-file output format (strict):
For each file, output exactly:

FILE: relative/path/to/file.ext
```language
<full file contents>
```

Rules:
- Always use Markdown code fences with language labels (python, javascript, etc.).
- Do not omit imports, helper functions, or constants.
- If modifying an existing file, output the full revised file (not a diff).
- Ensure consistency across files if dependencies exist.

Self-Check Before Finalizing (write it as bullets at the end):
- Syntax errors?
- All imports present?
- MWE runs?
- Types & docstrings included?
- For ML: proper split, seed, metric?

Modes:
#mode: write → produce code.
#mode: review → critique and suggest improvements.
#mode: explain → explain line by line or conceptually.

If you are unsure about domain facts, say 'I don't know' rather than inventing details.
"""
# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, static_folder="../static", template_folder="../templates")

# ---------------------------
# SQLite memory
# ---------------------------
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
    rows = cur.fetchall()[::-1]
    return [{"role": r[0], "content": r[1]} for r in rows]
# ---------------------------
# Helpers 
# ---------------------------

def project_base_dir(project: str) -> str:
    base = os.path.abspath(PROJECTS_DIR)
    path = os.path.abspath(os.path.join(base, project))
    if not path.startswith(base + os.sep) and path != base:
        abort(400, description="Invalid project path")
    os.makedirs(path, exist_ok=True)
    return path

def _is_allowed_file(path: str) -> bool:
    return "." in path and path.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _sanitize_rel_path(rel_path: str) -> Optional[str]:
    rel_path = rel_path.strip().replace("\\", "/")
    rel_path = re.sub(r"^/+","", rel_path)  # strip leading slashes
    if not _is_allowed_file(rel_path):
        return None
    # Prevent directory traversal
    if ".." in rel_path.split("/"):
        return None
    return rel_path

def _rank_project_files(file_paths: List[str]) -> List[str]:
    # Rank by (1) recency, (2) path depth (shallower first), (3) name preference
    def key(p: str):
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = 0
        depth = p.count(os.sep)
        name_bonus = 0
        basename = os.path.basename(p).lower()
        if basename in {"readme.md","requirements.txt","pyproject.toml","setup.py"}:
            name_bonus = 10
        return (-mtime, depth, -name_bonus)
    return sorted(file_paths, key=key)

def load_project_files_context(project: str) -> str:
    base_dir = project_base_dir(project)
    files_data = []
    byte_budget = MAX_PROMPT_CHARS

    preferred_globs = [
        "**/*.py", "**/*.ipynb", "**/*.md",
        "**/*.js", "**/*.ts", "**/*.tsx", "**/*.jsx",
        "**/*.json", "**/*.yml", "**/*.yaml",
        "**/*.toml", "**/*.ini",
        "**/*.html", "**/*.css",
        "README.md", "requirements.txt", "pyproject.toml", "setup.py"
    ]

    candidate_paths = set()
    for pattern in preferred_globs:
        candidate_paths.update(glob.glob(os.path.join(base_dir, pattern), recursive=True))

    ranked = _rank_project_files([p for p in candidate_paths if os.path.isfile(p)])

    count = 0
    for path in ranked:
        if count >= MAX_FILES_IN_CONTEXT:
            break
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
        block = f"FILE: {rel_path}\n```\n{content}\n```"
        # enforce overall char budget
        if len(block) > byte_budget:
            break
        files_data.append(block)
        byte_budget -= len(block)
        count += 1

    if not files_data:
        return "No existing project files found."
    return "\n\n".join(files_data)

FILE_BLOCK_RE = re.compile(
    r"FILE:\s*(?P<path>[^\n\r]+)\s*```(?P<lang>[\w.+-]+)?\s*\n(?P<code>.*?)```",
    re.DOTALL
)

CODE_FENCE_RE = re.compile(
    r"```(?P<lang>[\w.+-]*)\s*\n(?P<code>.*?)```",
    re.DOTALL
)

def _ensure_mode_header(text: str, default_mode: str = "write") -> str:
    first = text.strip().splitlines()[0].strip() if text.strip().splitlines() else ""
    if not re.search(r"^#mode:\s*(write|review|explain)\s*$", first, re.IGNORECASE):
        text = f"#mode: {default_mode}\n\n" + text
    return text

def _ensure_plan_section(text: str) -> str:
    if re.search(r"(?im)^\s*plan\s*$", text):
        return text
    # If no "Plan" heading, prepend a placeholder section to enforce habits.
    preface = (
        "Plan\n"
        "- Outline steps briefly.\n"
        "- Write complete code using the Multi-file format.\n"
        "- Add a tiny MWE when appropriate.\n"
        "- Include Self-Check at the end.\n\n"
    )
    return text if "Plan" in text[:400] else preface + text

def _wrap_lonely_fence_as_file(text: str) -> str:
    # If there are no FILE blocks but there is at least one code fence, wrap the first as scratch file.
    if FILE_BLOCK_RE.search(text):
        return text
    m = CODE_FENCE_RE.search(text)
    if not m:
        return text
    lang = (m.group("lang") or "text").lower()
    default_map = {
        "python": "scratch/main.py",
        "py": "scratch/main.py",
        "javascript": "scratch/index.js",
        "js": "scratch/index.js",
        "typescript": "scratch/index.ts",
        "ts": "scratch/index.ts",
        "json": "scratch/data.json",
        "html": "scratch/index.html",
        "css": "scratch/styles.css",
        "md": "scratch/README.md",
    }
    rel = default_map.get(lang, "scratch/snippet.txt")
    code = m.group("code")
    file_block = f"FILE: {rel}\n```{lang}\n{code}\n```"
    start, end = m.span()
    return text[:start] + file_block + text[end:]

def enforce_response_contract(text: str, default_mode: str = "write") -> str:
    text = _ensure_mode_header(text, default_mode=default_mode)
    text = _ensure_plan_section(text)
    text = _wrap_lonely_fence_as_file(text)
    # Normalize unlabeled fences to text to reduce parser misses
    text = re.sub(r"```(\s*\n)", "```text\1", text)
    return text

def save_generated_files(project: str, assistant_text: str) -> List[str]:
    base_dir = project_base_dir(project)
    matches = list(FILE_BLOCK_RE.finditer(assistant_text))
    saved: List[str] = []
    for m in matches:
        rel_path_raw = m.group("path")
        lang = (m.group("lang") or "").strip().lower()
        code = m.group("code")

        rel_path = _sanitize_rel_path(rel_path_raw)
        if not rel_path:
            # Skip dangerous or disallowed paths
            continue

        abs_path = os.path.abspath(os.path.join(base_dir, rel_path))
        if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
            continue

        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        # If fence had no language, try infer from extension
        if not lang:
            ext = rel_path.rsplit(".", 1)[-1].lower()
            lang = ext

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(code.strip())
        saved.append(rel_path)
    return saved

def parse_user_mode(text: str) -> str:
    m = re.search(r"#mode:\s*(write|review|explain)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # heuristics
    if re.search(r"explain|walk me through|what does.*mean", text, re.IGNORECASE):
        return "explain"
    if re.search(r"review|critique|improve|refactor", text, re.IGNORECASE):
        return "review"
    return "write"

def build_full_prompt(project: str, user_text: str) -> Tuple[str, str]:
    hist = load_recent(project, limit=8)
    hist_lines = [f"{m['role'].upper()}: {m['content']}" for m in hist]
    files_context = load_project_files_context(project)
    mode = parse_user_mode(user_text)

    planning_instructions = f"""
IMPORTANT:
- Always start with '#mode: {mode}' on the first line.
- Then write a 'Plan' section (3–7 bullets).
- Use the strict Multi-file format for ALL code.
- End with a 'Self-Check' bullet list.
"""

    sections = [
        SYSTEM_PROMPT.strip(),
        "\n--- Session Settings ---\n",
        f"Model: {MODEL}\nTemperature: {TEMPERATURE}\nTop_p: {TOP_P}\nSeed: {SEED}\n",
        "\n--- Project Context ---\n",
        files_context,
        "\n--- Conversation (recent) ---\n",
        "\n".join(hist_lines),
        "\n--- New Request ---\n",
        f"USER: {user_text}\n{planning_instructions}\nASSISTANT:"
    ]
    prompt = "\n".join(s for s in sections if s is not None)
    # enforce global char budget
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[-MAX_PROMPT_CHARS:]
    return prompt, mode
# ---------------------------
# Routes
# ---------------------------
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

@app.route("/settings", methods=["GET", "POST"])
def settings():
    """Get or update runtime settings without redeploy."""
    global MODEL, TEMPERATURE, TOP_P, NUM_CTX, SEED
    if request.method == "POST":
        data = request.json or {}
        MODEL = data.get("model", MODEL)
        TEMPERATURE = float(data.get("temperature", TEMPERATURE))
        TOP_P = float(data.get("top_p", TOP_P))
        NUM_CTX = int(data.get("num_ctx", NUM_CTX))
        SEED = int(data.get("seed", SEED))
    return jsonify({
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "num_ctx": NUM_CTX,
        "seed": SEED
    })

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
    full_prompt, mode = build_full_prompt(project, user_text)

    try:
        resp = ollama.generate(
            model=MODEL,
            prompt=full_prompt,
            stream=False,
            options={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": SEED,
                "num_ctx": NUM_CTX,
                "stop": ["\nUSER:", "\nSYSTEM:"],  # avoid the model continuing the prompt
            },
        )
        assistant_text_raw = resp.get("response") or resp.get("text") or ""
        assistant_text = enforce_response_contract(assistant_text_raw, default_mode=mode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Save message and any files that pass validation
    save_message(project, "assistant", assistant_text)
    saved_files = save_generated_files(project, assistant_text)

    return jsonify({"response": assistant_text, "saved_files": saved_files})

@app.route("/stream", methods=["POST"])
def stream():
    data = request.json or {}
    project = data.get("project", "default")
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return Response("data: ERROR: empty message\n\n", mimetype="text/event-stream")

    save_message(project, "user", user_text)
    full_prompt, mode = build_full_prompt(project, user_text)

    def generate():
        try:
            stream = ollama.generate(
                model=MODEL,
                prompt=full_prompt,
                stream=True,
                options={
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "seed": SEED,
                    "num_ctx": NUM_CTX,
                    "stop": ["\nUSER:", "\nSYSTEM:"],
                },
            )
            assistant_text_raw = ""
            for chunk in stream:
                token = chunk.get("response") or ""
                if token:
                    assistant_text_raw += token
                    safe_token = token.replace("\r", "").replace("\n", "\\n")
                    yield f"data: {safe_token}\n\n"

            assistant_text = enforce_response_contract(assistant_text_raw, default_mode=mode)
            save_message(project, "assistant", assistant_text)
            saved_files = save_generated_files(project, assistant_text)

            yield f"data: {{\"saved_files\": {json.dumps(saved_files)} }}\n\n"
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        save_message(project, "system", f"File uploaded: {filename}")
        return jsonify({"status": "ok", "filename": filename})
    else:
        return jsonify({"error": "file type not allowed"}), 400

@app.route("/delete_project", methods=["POST"])
def delete_project():
    data = request.json or {}
    project = (data.get("project") or "").strip()
    if not project:
        return jsonify({"error": "empty project name"}), 400

    # Delete from DB
    cur.execute("DELETE FROM messages WHERE project=?", (project,))
    conn.commit()

    # Delete project folder from disk
    project_dir = os.path.join(PROJECTS_DIR, project)
    if os.path.exists(project_dir):
        import shutil
        shutil.rmtree(project_dir)

    return jsonify({"status": "ok", "project": project})

@app.route("/cancel", methods=["POST"])
def cancel():
    try:
        ollama.cancel(model=MODEL)
        return jsonify({"status": "cancelled"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ---------------------------
# Optional: local lint & run (guarded)
# ---------------------------
def _run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 20) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)

@app.route("/run/<project>", methods=["POST"])
def run_project(project):
    if not RUNNER_ENABLED:
        return jsonify({"error": "runner disabled; set RUNNER_ENABLED=1"}), 400
    data = request.json or {}
    entry = (data.get("entry") or "main.py").strip()
    base = project_base_dir(project)
    if not _is_allowed_file(entry):
        return jsonify({"error": "disallowed entry point"}), 400
    path = os.path.join(base, entry)
    if not os.path.exists(path):
        return jsonify({"error": f"missing entry: {entry}"}), 404
    code, out, err = _run_cmd(["python", entry], cwd=base, timeout=60)
    return jsonify({"code": code, "stdout": out, "stderr": err})

@app.route("/lint/<project>", methods=["POST"])
def lint(project):
    if not LINTER_ENABLED:
        return jsonify({"error": "linter disabled; set LINTER_ENABLED=1"}), 400
    base = project_base_dir(project)
    # Try ruff, fallback to pyflakes
    try:
        import shutil as _shutil  
        has_ruff = _shutil.which("ruff") is not None
    except Exception:
        has_ruff = False
    if has_ruff:
        code, out, err = _run_cmd(["ruff", "."], cwd=base, timeout=60)
    else:
        code, out, err = _run_cmd(["python", "-m", "pyflakes", "."], cwd=base, timeout=60)
    return jsonify({"code": code, "stdout": out, "stderr": err})

if __name__ == "__main__":
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)


