const chatDiv = document.getElementById("chat");
const promptInput = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const stopBtn = document.getElementById("stop");
const clearBtn = document.getElementById("clear");
const projectSelect = document.getElementById("project");
const addProjectBtn = document.getElementById("addProject");
const deleteProjectBtn = document.getElementById("deleteProject");

let currentAbortController = null;

// ---------- Chat message helpers ----------
function makeUserNode(text) {
  const node = document.createElement("div");
  node.className = "message user";
  node.innerHTML = `<strong>You:</strong><br>${marked.parse(text)}`;
  return node;
}

function makeAssistantNode() {
  const node = document.createElement("div");
  node.className = "message assistant";
  node.innerHTML = `<strong>Assistant:</strong><br><em>...</em>`;
  return node;
}

function appendAndScroll(node) {
  chatDiv.appendChild(node);
  chatDiv.scrollTop = chatDiv.scrollHeight;
}

// ---------- Project management ----------
async function refreshProjects() {
  const res = await fetch("/projects");
  const data = await res.json();
  projectSelect.innerHTML = "";
  data.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p;
    projectSelect.appendChild(opt);
  });
}

addProjectBtn.addEventListener("click", async () => {
  const name = prompt("New project name:");
  if (!name) return;
  await fetch("/add_project", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project: name })
  });
  await refreshProjects();
  projectSelect.value = name;
});

deleteProjectBtn.addEventListener("click", async () => {
  const name = projectSelect.value;
  if (!name) return;
  if (!confirm(`Delete project "${name}"? This cannot be undone.`)) return;
  await fetch("/delete_project", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project: name })
  });
  await refreshProjects();
});

// ---------- Chat ----------
sendBtn.addEventListener("click", async () => {
  const prompt = promptInput.value.trim();
  const project = projectSelect.value || "default";
  if (!prompt) return;

  promptInput.value = "";
  appendAndScroll(makeUserNode(prompt));
  const assistantNode = makeAssistantNode();
  appendAndScroll(assistantNode);

  let fullText = "";
  currentAbortController = new AbortController();

  let finalPrompt = prompt;

  // --- Web search integration ---
  const useWeb = document.getElementById("useWeb").checked;
  if (useWeb) {
    try {
      const res = await fetch("/search_web", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: prompt })
      });
      const data = await res.json();
      if (data.results && data.results.length > 0) {
        const webText = data.results.slice(0, 5).join("\n- ");
        finalPrompt = `${prompt}\n\nWeb search results:\n- ${webText}`;
      }
    } catch (err) {
      console.error("Web search failed:", err);
    }
  }
  // --- End Web search ---

  fetch("/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: finalPrompt, project }),
    signal: currentAbortController.signal
  })
  .then(response => {
    if (!response.ok) {
      assistantNode.textContent = `Error: ${response.status}`;
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    function readLoop() {
      reader.read().then(({ done, value }) => {
        if (done) {
          hljs.highlightAll();
          return;
        }
        const chunkText = decoder.decode(value, { stream: true });
        const parts = chunkText.split("\n");
        for (const line of parts) {
          if (line.startsWith("data: ")) {
            const raw = line.slice(6);
            if (raw === "[DONE]") continue;
            if (raw.startsWith("ERROR:")) {
              assistantNode.textContent = raw;
              continue;
            }
            const formatted = raw.replace(/\\n/g, "\n");
            fullText += formatted;
            try {
              assistantNode.innerHTML = `<strong>Assistant:</strong><br>` + marked.parse(fullText);
              hljs.highlightAll();
            } catch {
              assistantNode.textContent = fullText;
            }
            chatDiv.scrollTop = chatDiv.scrollHeight;
          }
        }
        readLoop();
      });
    }
    readLoop();
  });
});

// ---------- Controls ----------
stopBtn.addEventListener("click", () => {
  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }
});

clearBtn.addEventListener("click", () => {
  chatDiv.innerHTML = "";
});

// ---------- Init ----------
refreshProjects();
