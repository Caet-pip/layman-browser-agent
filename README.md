# layman-browser-agent

> Early-stage project, very much a work in progress.

An autonomous browser agent that controls a real browser, uses an LLM to decide what to do, and has a built-in judge that checks whether the answer is good enough before returning it. Give it a task in plain English and it handles the browsing.

---

## How it works

### Browser layer — CDP over Playwright

The agent uses **raw CDP (Chrome DevTools Protocol)** for all page interaction. Playwright is used only to launch Chrome and manage the browser process. Everything else goes through CDP directly:

- **Snapshot** — `Accessibility.getFullAXTree` pulls the full accessibility tree and serializes it into a numbered list of interactive elements: `[1] button "Add to Cart"`, `[2] link "Nike Air Max"` etc. This is what the LLM sees instead of raw HTML.
- **Click** — `DOM.resolveNode` + `getBoundingClientRect()` scrolls the element into view and gets its viewport coordinates, then `Input.dispatchMouseEvent` fires the click. If the element has no visual box (screen-reader-only links), falls back to extracting the `href` and navigating directly.
- **Type** — CDP click to set keyboard focus, then `page.keyboard.type()`.
- **Navigate** — `page.goto()` via Playwright, CDP session reattached after.

### LLM layer

The agent sends the serialized AX snapshot to the LLM with a list of available tools (`browser_snapshot`, `browser_click`, `browser_type`, `browser_navigate`, `browser_scroll`, `browser_press_key`, `browser_tabs`, `browser_take_screenshot`). The LLM decides what to do next and the agent executes it in a loop.

Supports **OpenAI** (gpt-4o default) and **Ollama** (local models).

### Context management

- Token-based rolling window — keeps the most recent messages within a budget
- Atomic tool call pairs — assistant message + all tool results committed together so the context is never in a broken state
- Screenshot checks after navigation and clicks so the LLM can verify where it landed

### Task modes

The agent detects the task type and applies a mode-specific prompt:

- **Shopping** — builds a comparison table (columns adapt to the goal: price/availability for price tasks, style notes for aesthetic tasks), visits stores one at a time, never constructs URLs from memory
- **Research** — visits 3+ sources, cross-references facts, scrolls through full pages

### Judge

After the agent produces a final answer, a separate LLM call evaluates whether it actually completed the task. If not, the agent gets feedback and tries again (up to 3 rounds).

### Visible mouse (optional)

A purple/lilac SVG arrow cursor is injected into the page via JavaScript. It scrolls to each element before clicking so the cursor is always on screen. Re-injected automatically after every page load via a `page.on("load")` listener.

---

## Requirements

- Python 3.12+
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
python main.py --backend openai
python main.py --backend ollama --model llama3.2
python main.py --backend openai --visible-mouse
```

**Options:**

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--backend` | `openai`, `ollama` | `ollama` | LLM backend |
| `--model` | any model name | backend default | Override model (e.g. `llama3.2`, `qwen2.5:7b`, `gpt-4o-mini`) |
| `--browser` | `cdp`, `direct` | `cdp` | Browser client (`cdp` recommended) |
| `--visible-mouse` | flag | off | Show purple cursor during clicks |

Then type your task:

```
Task: find me the best budget running shoes under $80
Task: what is the best price for Assassin's Creed 3 on Nintendo Switch
Task: what are the pros and cons of a standing desk
```

Type `exit` to quit.

---

## Project structure

```
agent.py              — main agent loop, context management, judge
cdp_browser_client.py — CDP browser client (snapshot, click, type, scroll)
main.py               — CLI entry point
direct_browser_client.py — Playwright-based alternative client
parse_logs.py         — reads context_window.json + state_messages.json for debugging
test_zappos.py        — demo: cursor hovers over products, scrolls, clicks one
```

---

## Status

Early prototype. Expect rough edges.
