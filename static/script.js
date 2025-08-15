(function () {
  const sendBtn = document.getElementById("send");
  const clearBtn = document.getElementById("clear");
  const promptInput = document.getElementById("prompt");
  const projectSelect = document.getElementById("project");
  const addProjectBtn = document.getElementById("addProject");
  const chatDiv = document.getElementById("chat");

  // Load projects from backend
  function loadProjects() {
    fetch("/projects")
      .then(res => res.json())
      .then(projects => {
        projectSelect.innerHTML = "";
        projects.forEach(p => {
          const opt = document.createElement("option");
          opt.value = p;
          opt.textContent = p;
          projectSelect.appendChild(opt);
        });
      })
      .catch(err => console.error("Failed to load projects:", err));
  }

  loadProjects();

  // Add new project (persist to backend)
  addProjectBtn.addEventListener("click", () => {
    const name = prompt("Enter new project name:");
    if (name && name.trim()) {
      const trimmed = name.trim();
      fetch("/add_project", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project: trimmed })
      })
        .then(res => res.json())
        .then(data => {
          if (data.error) {
            alert("Error: " + data.error);
          } else {
            loadProjects();
            projectSelect.value = trimmed;
          }
        })
        .catch(err => {
          alert("Failed to add project: " + err);
        });
    }
  });

  // Append and scroll helper
  function appendAndScroll(node) {
    chatDiv.appendChild(node);
    chatDiv.scrollTop = chatDiv.scrollHeight;
  }

  function makeUserNode(text) {
    const d = document.createElement("div");
    d.className = "user";
    d.textContent = text;
    return d;
  }

  function makeAssistantNode() {
    const d = document.createElement("div");
    d.className = "assistant";
    d.style.whiteSpace = "pre-wrap";
    return d;
  }

  sendBtn.addEventListener("click", () => {
    const prompt = promptInput.value.trim();
    const project = projectSelect.value || "default";
    if (!prompt) return;

    promptInput.value = "";

    appendAndScroll(makeUserNode(prompt));
    const assistantNode = makeAssistantNode();
    appendAndScroll(assistantNode);

    let fullText = "";

    fetch("/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: prompt, project })
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
                  assistantNode.innerHTML = marked.parse(fullText);
                  hljs.highlightAll();
                } catch {
                  assistantNode.textContent = fullText;
                }
                chatDiv.scrollTop = chatDiv.scrollHeight;
              }
            }
            readLoop();
          }).catch(err => {
            assistantNode.textContent = "Stream error: " + err;
          });
        }
        readLoop();
      })
      .catch(err => {
        assistantNode.textContent = "Fetch error: " + err;
      });
  });

  clearBtn.addEventListener("click", () => {
    chatDiv.innerHTML = "";
  });

  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      sendBtn.click();
    }
  });
})();

async function uploadFile(project) {
    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Select a file first");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const res = await fetch(`/upload_file/${encodeURIComponent(project)}`, {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    if (data.status === "ok") {
        alert(`Uploaded: ${data.filename}`);
        loadProjects(); // refresh project list
    } else {
        alert("Error: " + data.error);
    }
}