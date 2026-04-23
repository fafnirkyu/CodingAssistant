const chatDiv = document.getElementById("chat");
const promptInput = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const stopBtn = document.getElementById("stop");
const clearBtn = document.getElementById("clear");
const projectSelect = document.getElementById("project");
const addProjectBtn = document.getElementById("addProject");
const deleteProjectBtn = document.getElementById("deleteProject");

let currentAbortController = null;

// ---------- NEW: Project History Loading ----------

async function loadHistory(project) {
  chatDiv.innerHTML = '<div class="message assistant"><em>Loading history...</em></div>';
  
  try {
    const res = await fetch(`/history/${encodeURIComponent(project)}`);
    const data = await res.json();
    
    chatDiv.innerHTML = ""; // Clear loading indicator

    if (data.history && data.history.length > 0) {
      data.history.forEach(msg => {
        if (msg.role === "user") {
          appendAndScroll(makeUserNode(msg.content));
        } else {
          // Assistant messages need markdown parsing
          const node = makeAssistantNode();
          node.innerHTML = `<strong>Assistant:</strong><br>${marked.parse(msg.content)}`;
          appendAndScroll(node);
        }
      });
      // Re-highlight all code blocks after loading
      if (typeof hljs !== 'undefined') hljs.highlightAll();
    } else {
      chatDiv.innerHTML = '<div class="message assistant"><em>New project started. No history found.</em></div>';
    }
  } catch (err) {
    console.error("Error loading history:", err);
    chatDiv.innerHTML = '<div class="message assistant"><em>Error loading history for this project.</em></div>';
  }
}

// Listen for dropdown changes
projectSelect.addEventListener("change", () => {
  loadHistory(projectSelect.value);
});

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
  const currentVal = projectSelect.value;
  
  projectSelect.innerHTML = "";
  data.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p;
    projectSelect.appendChild(opt);
  });

  // Keep selection if it still exists, otherwise load the first project
  if (data.includes(currentVal)) {
    projectSelect.value = currentVal;
  } else if (data.length > 0) {
    projectSelect.value = data[0];
    loadHistory(data[0]); // Load history for the initial project
  }
}

addProjectBtn.addEventListener("click", async () => {
  const name = prompt("Project Name:");
  if (!name) return;
  await fetch(`/add_project/${name}`, { method: "POST" });
  await refreshProjects();
});

deleteProjectBtn.addEventListener("click", async () => {
  const p = projectSelect.value;
  if (!p || !confirm(`Delete project ${p}?`)) return;
  await fetch(`/delete_project/${p}`, { method: "DELETE" });
  await refreshProjects();
});

// ---------- Core Chat Logic ----------
sendBtn.addEventListener("click", async () => {
  const text = promptInput.value.trim();
  if (!text) return;

  const project = projectSelect.value;
  const useWeb = document.getElementById("useWeb").checked;

  appendAndScroll(makeUserNode(text));
  promptInput.value = "";

  const assistantNode = makeAssistantNode();
  appendAndScroll(assistantNode);

  currentAbortController = new AbortController();

  try {
    let search_results = [];
    if (useWeb) {
      assistantNode.innerHTML = `<strong>Assistant:</strong><br><em>Searching web...</em>`;
      const sResp = await fetch("/search_web", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text })
      });
      const sData = await sResp.json();
      search_results = sData.results || [];
    }

    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project, message: text, search_results }),
      signal: currentAbortController.signal
    });

    const data = await res.json();
    if (data.response) {
      assistantNode.innerHTML = `<strong>Assistant:</strong><br>${marked.parse(data.response)}`;
      if (typeof hljs !== 'undefined') hljs.highlightAll();
    } else if (data.error) {
      assistantNode.textContent = "Error: " + data.error;
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      assistantNode.innerHTML += "<br><em>[Stopped]</em>";
    } else {
      assistantNode.textContent = "Error: " + err.message;
    }
  } finally {
    currentAbortController = null;
    chatDiv.scrollTop = chatDiv.scrollHeight;
  }
});

// Clear UI only (doesn't delete database)
clearBtn.addEventListener("click", () => {
  chatDiv.innerHTML = "";
});

async function uploadFile(project) {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) {
    alert("Please select a file first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res = await fetch(`/upload_file/${project}`, { method: "POST", body: formData });
    const data = await res.json();
    if (data.status === "ok") {
      alert(`Uploaded: ${data.filename}`);
      fileInput.value = "";
    } else {
      alert(`Failed: ${data.error}`);
    }
  } catch (err) {
    alert("Error uploading file.");
  }
}

// Ctrl+Enter support
promptInput.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") sendBtn.click();
});

// Initial Init
refreshProjects();