"""
server.py

Web UI server for the browser agent.
One long-running agent instance — the web UI connects to it via websocket.

Run with:
    python server.py --backend openai
    python server.py --backend ollama --model gemma4:4b
"""
import asyncio
import argparse
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from agent import BrowserAgent

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Single shared agent instance ──────────────────────────────────────────────

_agent: BrowserAgent | None = None
_agent_lock = asyncio.Lock()   # one task at a time
_agent_busy = False


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _agent_busy
    await websocket.accept()

    # Queue for human answers coming in while the agent is running
    answer_queue: asyncio.Queue = asyncio.Queue()

    async def run_in_background(task: str):
        global _agent_busy
        async with _agent_lock:
            _agent_busy = True
            try:
                await _run_task(task, websocket, answer_queue)
            finally:
                _agent_busy = False

    try:
        while True:
            data = await websocket.receive_json()

            # Human answered a follow-up question — unblock the waiting agent
            if data.get("type") == "human_answer":
                await answer_queue.put(data.get("text", ""))
                continue

            task = data.get("task", "").strip()
            if not task:
                continue

            if _agent_busy:
                await websocket.send_json({"type": "error", "text": "Agent is busy — wait for current task to finish."})
                continue

            # Run agent as a background task so this loop stays free to receive human_answer messages
            asyncio.create_task(run_in_background(task))

    except WebSocketDisconnect:
        if _agent:
            _agent.on_card = None
            _agent.on_thinking = None
            _agent.on_ask_human = None
        _agent_busy = False


async def _run_task(task: str, websocket: WebSocket, answer_queue: asyncio.Queue):
    """Wire up callbacks then run the agent task, streaming events to the websocket."""
    agent = _agent

    # ── Callbacks that push events to this websocket ──
    async def send_card(card: dict):
        await websocket.send_json({"type": "product_card", **card})

    async def send_thinking(text: str):
        await websocket.send_json({"type": "thinking", "text": text})

    async def ask_human(question: str) -> str:
        await websocket.send_json({"type": "ask_human", "question": question})
        return await answer_queue.get()

    agent.on_card = send_card
    agent.on_thinking = send_thinking
    agent.on_ask_human = ask_human

    # Detect mode and set it
    if agent.is_continuation(task):
        await websocket.send_json({"type": "thinking", "text": "Continuing previous task..."})
    else:
        mode = agent.detect_mode(task)
        agent.set_mode(mode)
        if mode:
            await websocket.send_json({"type": "thinking", "text": f"Mode: {mode}"})

    try:
        result = await agent.run(task)
        await websocket.send_json({"type": "done", "text": result})
    except Exception as e:
        await websocket.send_json({"type": "error", "text": str(e)})
    finally:
        agent.on_card = None
        agent.on_thinking = None
        agent.on_ask_human = None


# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _agent
    _agent = BrowserAgent(
        backend=_args.backend,
        model=_args.model,
        browser=_args.browser,
        visible_mouse=_args.visible_mouse,
    )
    await _agent.connect()
    print("[Server] Agent ready. Open http://localhost:8000")


@app.on_event("shutdown")
async def shutdown():
    if _agent:
        await _agent.close()


# ── Entry point ───────────────────────────────────────────────────────────────

_args = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--browser", choices=["cdp", "direct"], default="cdp")
    parser.add_argument("--visible-mouse", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    _args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=_args.port)
