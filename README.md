# layman-browser-agent

> Early-stage project, very much a work in progress.

An autonomous browser agent that can navigate the web, search for products, do research, and answer questions — all on its own. Give it a task in plain English and it handles the browsing.

---

## How it works

The agent controls a real browser (via Playwright or MCP), uses an LLM to decide what to do next, and has a built-in judge that checks whether the final answer is actually good enough before returning it.

---

## Requirements

- Python 3.12+
- Node.js (for MCP browser servers)
- An OpenAI API key, or Ollama running locally

---

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

Set your OpenAI key if using OpenAI:

```bash
export OPENAI_API_KEY=your_key_here
```

---

## Running

```bash
python main.py --backend openai --browser direct
```

**Options:**

- `--backend` — `openai` or `ollama` (default: ollama)
- `--browser` — `direct` (persistent Playwright), `playwright` (MCP), or `cdp` (Chrome DevTools Protocol)

Then just type your task:

```
Task: find me the best budget running shoes under $80
Task: what are the pros and cons of a standing desk
```

Type `exit` to quit.

---

## Status

This is an early prototype. Expect rough edges.
