# browser-agent

> Early-stage project, very much a work in progress.

An autonomous browser agent with a web UI. Give it a task in plain English — it browses the web, makes decisions, and streams results back to you in real time as animated product cards. It asks you clarifying questions before searching, researches trends first, then hunts across sites organically.

---

## How it works

### Browser layer — CDP over Playwright

The agent controls a real Chrome browser using **raw CDP (Chrome DevTools Protocol)**. Playwright is used only to launch the browser process.

- **Snapshot** — `Accessibility.getFullAXTree` serializes the page into a numbered list of interactive elements: `[1] button "Add to Cart"`, `[2] link "Nike Air Max"`. This is what the LLM sees instead of raw HTML.
- **Click** — `DOM.resolveNode` + `getBoundingClientRect()` scrolls the element into view and gets its viewport coordinates. `Input.dispatchMouseEvent` fires the click. Falls back to `href` navigation if the element has no visual box.
- **Type** — CDP click to focus, then `page.keyboard.type()`.
- **Screenshot** — taken on demand by the LLM when visual confirmation is needed. Described in one sentence and returned as text to avoid bloating context.

### LLM layer

The agent sends the AX snapshot to the LLM with available tools. The LLM decides what to do next and the agent executes it in a loop.

Supports **OpenAI** (gpt-4.1 default) and **Ollama** (local models, gemma4:31b default).

### Context management

- Token-based rolling window — keeps messages within an 80k token budget
- Stale DOM snapshots summarized to URL + title only — prevents quadratic context bloat
- Atomic tool call pairs — assistant + all tool results committed together, never a broken pair
- `_heal_messages()` strips incomplete pairs at the start of each step

### Real-time product cards

When the agent finds a product that genuinely fits the request, it calls `emit_product_card` — a tool that fires the card to the browser UI immediately via WebSocket. The UI shows it as an animated floating card without waiting for the task to finish.

- Cards are placed randomly within the visible screen area with collision detection
- Cards bob up and down with staggered animation phases
- Hover to expand — image grows, price and details reveal below
- Click to open the product URL

### Ask-human flow

Before searching, the agent asks 1–2 clarifying questions (style, budget, occasion, etc.) directly in the UI via a frosted glass prompt. Answers go back to the agent over the same WebSocket connection without blocking the receive loop.

### Shopping mode

- Research first — searches for what's trending or well-reviewed before picking sites
- Chooses which sites to visit organically based on the request — no hardcoded store list
- Only emits cards for products that genuinely fit the user's stated preferences
- Duplicate URL tracking prevents the same product being emitted twice

### Judge

After the agent produces a final answer, a separate LLM call checks whether the task was actually completed. If not, the agent gets feedback and tries again (up to 3 rounds). The final summary is accessible by clicking the thinking bar at the bottom — not dumped on screen.

### Visible mouse

A purple SVG cursor is injected into the page. It moves to each element before clicking with a 1.4s pause so you can see what it's about to do. Re-injected after every page load. Mouse coordinates logged to the terminal for debugging.

---

## Requirements

- Python 3.12+
- An OpenAI API key, or Ollama running locally
- `uvicorn[standard]` for WebSocket support

---

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

Set your OpenAI key if using OpenAI:

```bash
export OPENAI_API_KEY=your_key_here
```

---

## Running

### Web UI (recommended)

```bash
python server.py --backend openai
python server.py --backend ollama --model gemma4:31b-cloud
python server.py --backend openai --visible-mouse
```

Then open **http://localhost:8000** in your browser.

### CLI

```bash
python main.py --backend openai
python main.py --backend ollama --model llama3.2
```

**Options:**

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--backend` | `openai`, `ollama` | `ollama` | LLM backend |
| `--model` | any model name | backend default | Override model |
| `--browser` | `cdp`, `direct` | `cdp` | Browser client |
| `--visible-mouse` | flag | off | Show cursor during clicks |
| `--port` | number | `8000` | Server port (web UI only) |

---

## Project structure

```
agent.py                  — agent loop, context management, judge, tool dispatch
cdp_browser_client.py     — CDP browser client (snapshot, click, type, scroll, cursor)
server.py                 — FastAPI WebSocket server for the web UI
static/index.html         — web UI (frosted glass, floating product cards)
main.py                   — CLI entry point
mcp_client.py             — alternative MCP-based browser client
direct_browser_client.py  — alternative Playwright-based browser client
```

---

## Status

Early prototype. Expect rough edges.
