import asyncio

from typing import List, Dict

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Import existing RAG function
from src.Retriever import Retriever
from src.Utility import setup_logger

app = FastAPI()
retriever = Retriever()
logger = setup_logger("Retriever.log")

# =====================================================
# Business Logic Layer (NO FastAPI decorators here)
# =====================================================
class ChatService:
    def __init__(self):
        self.chat_history: List[Dict] = []
        self.lock = asyncio.Lock()
        logger.info("Starting Chat Session module.")


    async def ask(self, prompt: str, trace_id: str) -> Dict:
        """
             Process a user prompt and return RAG response.
             Handles both async and sync retriever.run safely.
        """
        trace_id = trace_id or "no-trace"

        logger.info("Received user prompt",
                    extra={"trace_id": trace_id, "prompt": prompt})

        # store user message
        self.chat_history.append({
            "role": "user",
            "message": prompt,
            "sources": []
        })

        # logger.info("User prompt - %s",prompt)

        # call RAG
        async with self.lock:
            try:
                # Detect if retriever.run is a coroutine
                if asyncio.iscoroutinefunction(retriever.run):
                    result = await retriever.run(prompt)
                else:
                    # Run sync function in threadpool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, retriever.run, prompt)

            except Exception as e:
                logger.exception("Retriever failed", extra={"trace_id": trace_id})
                result = {"answer": "Sorry, I couldn't get an answer right now.", "sources": []}

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # store bot response
        self.chat_history.append({
            "role": "r2d2",
            "message": answer,
            "sources": sources # getting sources but it is suppressed in front end
        })

        MAX_HISTORY = 100
        if len(self.chat_history) > MAX_HISTORY:
            self.chat_history = self.chat_history[-MAX_HISTORY:]

        logger.info(
            "RAG response generated",
            extra={"trace_id": trace_id or "no-trace", "sources": sources}
        )

        return {
            "answer": answer,
            "sources": sources
        }


    def get_history(self):
        # return self.chat_history[-500:]
        return [
            {
                "role": msg["role"],
                "message": msg["message"],
                "sources": []  # hide sources
            }
            for msg in self.chat_history[-50:]
        ]


# Singleton service instance
chat_service = ChatService()


# =====================================================
# API Layer
# =====================================================
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>GenAI-ChatBot</title>
    <style>
        body {
            font-family: Arial;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
            background-color: #f7f7f7;
        }
        #leftPanel {
            width: 30%;
            padding: 20px;
            border-right: 1px solid #ccc;
            box-sizing: border-box;
            background-color: #ffffff;
        }
        #rightPanel {
            width: 70%;
            padding: 20px;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            box-sizing: border-box;
            background-color: #eef2f5;
        }
        h3 {
            margin-top: 0;
        }
        textarea {
            width: 100%;
            height: 100px;
            resize: none;
            padding: 10px;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        button {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: none;
            background-color: #007bff;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .chat-message {
            max-width: 60%;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 20px;
            word-wrap: break-word;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            font-size: 14px;
        }
        .user {
            background-color: #daf1ff;
            align-self: flex-start;
            border-bottom-left-radius: 0;
        }
        .bot {
            background-color: #f0f0f0;
            align-self: flex-end;
            border-bottom-right-radius: 0;
        }
        .thinking {
            font-style: italic;
            color: #555;
        }
        #rightPanel::-webkit-scrollbar {
            width: 8px;
        }
        #rightPanel::-webkit-scrollbar-thumb {
            background-color: #ccc;
            border-radius: 4px;
        }
    </style>
</head>
<body>

<div id="leftPanel">
    <h3>How May I Help You ?</h3>
    <form id="chatForm">
        <textarea id="prompt" placeholder="Type your question..."></textarea>
        <button type="submit">Send</button>
    </form>
</div>

<div id="rightPanel">
    <!-- Chat messages will appear here -->
</div>

<script>
const chatContainer = document.getElementById("rightPanel");

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

async function refreshChat() {
    try {
        const res = await fetch("/history", {cache: "no-store"});
        const data = await res.json();

        chatContainer.innerHTML = "";

        data.history.forEach(msg => {
            const div = document.createElement("div");
            div.className = "chat-message " + (msg.role === "r2d2" ? "bot" : "user");
            const displayName = msg.role === "r2d2" ? "R2D2" : "USER";
            div.innerHTML = `<strong>${displayName}:</strong><br>${escapeHtml(msg.message)}`;
            chatContainer.appendChild(div);
        });

        chatContainer.scrollTop = chatContainer.scrollHeight;
    } catch (err) {
        console.error("Failed to fetch chat history:", err);
    }
}

document.getElementById("chatForm").onsubmit = async function(e) {
    e.preventDefault();
    const promptInput = document.getElementById("prompt");
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    // Show user message immediately
    const userDiv = document.createElement("div");
    userDiv.className = "chat-message user";
    userDiv.innerHTML = `<strong>You:</strong><br>${escapeHtml(prompt)}`;
    chatContainer.appendChild(userDiv);

    // Show thinking indicator
    const thinking = document.createElement("div");
    thinking.id = "thinking";
    thinking.className = "chat-message bot thinking";
    thinking.innerHTML = "R2D2 is thinking...";
    chatContainer.appendChild(thinking);

    chatContainer.scrollTop = chatContainer.scrollHeight;

    const formData = new FormData();
    formData.append("prompt", prompt);

    try {
        const res = await fetch("/ask", { method: "POST", body: formData });
        if (!res.ok) throw new Error("Backend error");

        const data = await res.json();
        const t = document.getElementById("thinking");
        if (t) t.remove();

        await refreshChat();
        promptInput.value = "";
    } catch (err) {
        console.error("Error sending prompt:", err);
        const t = document.getElementById("thinking");
        if (t) t.remove();
        alert("Failed to get response from backend.");
    }
};

// Initial load
refreshChat();
</script>

</body>
</html>
"""

@app.post("/ask")
async def ask(request: Request,prompt: str = Form(...)):
    logger.info(f"prompt: {prompt}")
    trace_id = getattr(request.state, "trace_id", "no-trace")
    try:
        result = await chat_service.ask(prompt,trace_id)
        return JSONResponse(result)

    except Exception as e:
        logger.exception(
            "Chat processing failed",
            extra={"trace_id": trace_id}
        )
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/history")
async def history():
    logger.debug(f"history")
    return JSONResponse({
        "history": chat_service.get_history()
    })
