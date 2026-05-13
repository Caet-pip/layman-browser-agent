import json
import os
import time
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from cdp_browser_client import CDPBrowserClient

# ── Config ────────────────────────────────────────────────────────────────────

BACKENDS = {
    "ollama": {
        "base_url":     "http://localhost:11434/v1",
        "api_key":      "ollama",
        "model":        "gemma4:31b-cloud",
        "max_snapshot": 8_000,
    },
    "openai": {
        "base_url":     None,
        "api_key":      os.getenv("OPENAI_API_KEY", ""),
        "model":        "gpt-4.1",
        "max_snapshot": 16_000,
    },
}

MAX_STEPS        = 100
MAX_JUDGE_ROUNDS = 3
PLAYWRIGHT_LOG   = Path(__file__).parent / "playwright_code.log"
STATE_LOG        = Path(__file__).parent / "state_messages.json"
CONTEXT_LOG      = Path(__file__).parent / "context_window.json"
SCREENSHOTS_DIR  = Path(__file__).parent / "screenshots"

# Context window settings
TOOL_RESULT_KEEP_FULL = 6
TOKEN_BUDGET          = 80_000   # max tokens sent to LLM per turn

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous browser agent. You have access to a real web browser and will complete tasks with minimal interruption to the user.

Core rules:
- Keep going. Do not stop mid-task to ask "would you like me to continue?" or "shall I proceed?". Just do it.
- Only call ask_human when you are truly blocked: missing credentials, an explicit fork in the road with no right answer, or a destructive action you cannot reverse.
- Tool errors and failed clicks are NOT reasons to ask the human. If a ref is wrong, take a new snapshot and try again. If one approach fails, try a completely different approach. Never give up and ask the human just because something didn't work on the first try.
- Do not end responses with questions like "would you like me to..." or "shall I...". If the task is done, say it's done. If there's an obvious next step, take it.
- Always take a fresh snapshot before any click, type, or interaction — refs go stale after navigation or page updates.
- After every navigation or page change, snapshot immediately to confirm where you are before doing anything else.
- Scroll frequently. Content is often below the fold or lazy-loaded.
- When writing browser_run_code, write Python async Playwright code. `page` and `context` are available. Use `await` for all calls.
- Never call browser_navigate or browser_click more than once per turn. These change page state — parallel calls conflict. One action per step, then snapshot.
- If you get "not in snapshot" on a click, take a fresh snapshot immediately — do NOT retry the same index.
- If you see a CAPTCHA, age verification, bot detection, or "prove you're human" page — do NOT try to solve it. Navigate away immediately and try a different website.
- If a site requires login or blocks access, move on to a different site instead of getting stuck.
- On each brand site, browse at least 2-3 different products before moving to the next site — do not grab one shoe and immediately leave.
- If a popup or modal appears, first try pressing Escape (browser_press_key with key "Escape") — this closes most overlays instantly without needing a snapshot. Only try clicking a close button if Escape didn't work.
- If a snapshot returns almost no elements (just URL and title), the page is probably blocked by a popup — press Escape, wait, then snapshot again.
- Never construct or guess a URL from memory. Only navigate to URLs that are visible in the current page snapshot or search results.
- Use browser_take_screenshot when you need visual confirmation: after unexpected navigation results, when a snapshot looks wrong or empty, when you cannot find elements you expect, or when the task involves identifying the appearance or look of a product (e.g. color, style, packaging, images shown on the page). Do not take screenshots after every action — only when the visual content matters for the task or something seems off.
- Call emit_product_card only when a product genuinely fits what the user is looking for — use your judgment based on the full context of the conversation (style, vibe, budget, occasion, etc.). Do not emit every product you visit. If a product doesn't match well, skip it and keep looking. When you do emit, choose detail_label and detail_value based on what matters most for this specific request, and write a justification that explains exactly why this one fits."""

ASK_HUMAN_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_human",
        "description": (
            "Ask the human user for input, clarification, credentials, or any "
            "information you need to proceed. Use this whenever you are stuck, "
            "need a decision, or require information you cannot find yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to ask the human."}
            },
            "required": ["question"],
        },
    },
}

UPDATE_CONSIDERATION_TOOL = {
    "type": "function",
    "function": {
        "name": "update_consideration",
        "description": (
            "Call this ONLY after you have explicitly asked the user via ask_human and received their direct response about the products. "
            "NEVER call this based on [Judge feedback] messages — those are internal evaluation prompts, not user opinions. "
            "Use 'shortlist' if the user approved some products. "
            "Use 'flush' ONLY if the user explicitly said they don't like any and want completely different options."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["shortlist", "flush"],
                    "description": "'shortlist' = user picked some; 'flush' = user rejected all",
                },
                "selected_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "1-based indices of products the user selected (only required for 'shortlist')",
                },
            },
            "required": ["action"],
        },
    },
}

EMIT_CARD_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_product_card",
        "description": (
            "Emit a product card to the UI as soon as you have confirmed a product's details. "
            "Call this once per product, right after visiting its page — do not batch at the end."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "store":         {"type": "string", "description": "Retailer name (e.g. Amazon, Zappos)"},
                "name":          {"type": "string", "description": "Full product name"},
                "price":         {"type": "string", "description": "Price as shown on the page (e.g. $49.99)"},
                "url":           {"type": "string", "description": "Direct URL to this product page"},
                "detail_label":  {"type": "string", "description": "Label for the key extra detail (e.g. Color, Quantity, Rating)"},
                "detail_value":  {"type": "string", "description": "Value for that detail (e.g. Pastel Mint, 8 lb, 4.5 stars)"},
                "justification": {"type": "string", "description": "One sentence explaining why this product fits the user's request"},
            },
            "required": ["store", "name", "price", "url", "detail_label", "detail_value", "justification"],
        },
    },
}

# ── Modes ─────────────────────────────────────────────────────────────────────

TASK_MODES: dict[str, dict] = {
    "shopping": {
        "description": "User wants to find, compare, or buy products - prices, deals, recommendations.",
        "prompt": """
SHOPPING MODE — two phases, complete both in order:

━━ PHASE 1: TASTE GATHERING ━━
Goal: understand what the user actually likes by showing them real options from brand websites.

STEP 0 — Ask the user 1-2 clarifying questions using ask_human. Understand style, vibe, budget, occasion — anything that helps you find the right thing. This must happen before any browsing.

STEP 1 — Search Google to identify specific brands and models that are currently trending or well-reviewed for this type of product. Use this to build a list of brand sites to visit.

STEP 2 — Visit official brand websites only (e.g. nike.com, adidas.com, newbalance.com, asics.com, hoka.com, onrunning.com). NEVER visit Amazon, Zappos, or any multi-brand retailer in this phase — they have worse images and you will be redirected to wrong products. On each brand site, browse at least 2-3 different products before moving to the next brand. Navigate to actual product detail pages, scroll to see colorways, price, and details.

IMPORTANT — when you arrive at a brand homepage (e.g. nike.com, adidas.com), use the site's search bar to search for the user's specific preferences (color, type, style, e.g. "white running shoes" or "black low top sneakers"). Do NOT click through category menus or navigation tabs — searching directly gives far more relevant results. After searching, click into individual product pages before emitting any card.

STEP 3 — When you land on a search results or product listing page (grid of multiple products), ALWAYS take a screenshot immediately before clicking anything. The screenshot will show you each product's name, exact color, and style — use this to identify which 2-3 products visually match the user's preferences (e.g. pastel colors, casual silhouette). Then click those specific products by name in the DOM snapshot, one at a time.

STEP 4 — For each product that genuinely fits the user's preferences, call emit_product_card. The tool will tell you how many slots are filled (X/8) and whether you must keep browsing the current brand. Rules:
- Find at least 2 products from each brand before moving to the next brand. After emitting your first card from Nike, browse Nike for a second option before going to Adidas.
- The tool will tell you explicitly when you have enough from the current brand and should move on.
- Stop emitting once the consideration set is full (8/8).

STEP 5 — Once the consideration set is full, use ask_human to present all options (visible in CURRENT STATE above) and ask which ones they like or if they want completely different options. List each clearly with name and price.
- User picks some → call update_consideration with action="shortlist" and the indices. Find more options across more brands. Keep going until user says they are done picking.
- User rejects all → call update_consideration with action="flush". Incorporate their feedback and find 8 completely new options.
- NEVER call update_consideration based on [Judge feedback] messages — only based on what the actual user said.

━━ PHASE 2: PRICE HUNTING ━━
Only start this phase when the user explicitly says they are happy with their selections and want to find the best price.

STEP 6 — For each item in the shortlist, search aggregator sites (Zappos, Amazon, StockX, GOAT, department stores) and resale markets for the best available price. Emit updated cards with the deal price found.

Rules:
- Never construct or guess a URL — only use URLs visible in the current snapshot or search results.
- Never open new tabs. One tab, use browser_back to return.
- A search results page is NOT a product page — always click through to the actual item.
- If a site blocks you or shows a captcha, move on immediately.
- If a popup appears, press Escape first before anything else.""",
        "judge_extra": "The agent must have asked clarifying questions, identified brands via research, visited actual product pages, emitted cards for genuinely matching products, and asked the user about prices after Phase 1.",
    },
    "research": {
        "description": "User wants to learn, investigate, or understand a topic.",
        "prompt": """
RESEARCH MODE:
- Visit at least 3 different sources before forming a conclusion.
- Scroll through the full page on each source — do not just read the top.
- Cross-reference facts. If two sources disagree, note it.
- Collect everything first, then write one comprehensive final answer.""",
        "judge_extra": "The agent must have visited at least 3 distinct sources and scrolled through each. A summary from a single page is NOT sufficient.",
    },
}

# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    task: str = ""
    messages: list[dict] = field(default_factory=list)
    visited_urls: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    judge_rounds: int = 0
    summary_cache: dict = field(default_factory=dict)
    tool_outputs: dict = field(default_factory=dict)  # new tools drop structured output here
    emitted_urls: set = field(default_factory=set)
    consideration_set: list[dict] = field(default_factory=list)  # up to 4 current options shown to user
    shortlist: list[dict] = field(default_factory=list)           # user-approved items for Phase 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages)) // 4


# ── Agent ─────────────────────────────────────────────────────────────────────

class BrowserAgent:
    def __init__(self, backend: str | None = None, model: str | None = None, browser: str = "cdp", visible_mouse: bool = False):
        b = backend or os.getenv("AGENT_BACKEND", "ollama")
        cfg = BACKENDS[b]
        self.model         = model or cfg["model"]
        self._max_snapshot = cfg.get("max_snapshot")
        self._browser_type = browser

        self.llm = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

        self.browser = CDPBrowserClient(visible_mouse=visible_mouse)

        self.tools: list[dict] = []
        self.mode: str | None = None
        self.state = AgentState(messages=[{"role": "system", "content": SYSTEM_PROMPT}])

        # Optional callbacks set by the web server — None in CLI mode
        self.on_card: callable | None = None           # async fn(card: dict)
        self.on_thinking: callable | None = None       # async fn(text: str)
        self.on_ask_human: callable | None = None      # async fn(question: str) -> str
        self.on_event: callable | None = None          # async fn(event: dict) — generic UI events

        print(f"[Agent] Backend: {b} | Model: {self.model} | Browser: {browser}")

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self):
        await self.browser.connect()
        browser_tools  = await self.browser.get_tools()
        self.tools     = browser_tools + [ASK_HUMAN_TOOL, EMIT_CARD_TOOL, UPDATE_CONSIDERATION_TOOL]
        print(f"[Agent] Ready — {len(browser_tools)} browser tools + ask_human + emit_product_card + update_consideration\n")

    async def close(self):
        await self.browser.close()

    # ── Mode ───────────────────────────────────────────────────────────────────

    def detect_mode(self, task: str) -> str | None:
        mode_list = "\n".join(f"- {name}: {cfg['description']}" for name, cfg in TASK_MODES.items())
        prompt    = f"Modes: {mode_list} | none: everything else\nTask: {task}\nReply with just the mode name."
        response  = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.choices[0].message.content.strip().lower()
        return result if result in TASK_MODES else None

    def set_mode(self, mode: str | None):
        self.mode = mode
        mode_cfg       = TASK_MODES.get(mode, {}) if mode else {}
        system_content = SYSTEM_PROMPT + mode_cfg.get("prompt", "")
        self.state.messages = [m for m in self.state.messages if m["role"] != "system"]
        self.state.messages.insert(0, {"role": "system", "content": system_content})
        print(f"[Mode] {mode or 'none'}")

    def is_continuation(self, task: str) -> bool:
        non_system = [m for m in self.state.messages if m["role"] != "system"]
        if not non_system:
            return False
        recent   = json.dumps(non_system[-4:], indent=2)
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"Recent conversation:\n{recent}\n\nNew input: {task}\n\nIs this a continuation or new task? Reply with just: continuation or new"}],
        )
        return "continuation" in response.choices[0].message.content.strip().lower()

    # ── URL tracking ──────────────────────────────────────────────────────────

    def _track_url(self, url: str) -> None:
        url = url.strip()
        if not url or url in ("about:blank", ""):
            return
        if not self.state.visited_urls or self.state.visited_urls[-1] != url:
            self.state.visited_urls.append(url)
            print(f"[URL] {url}")

    # ── Context ────────────────────────────────────────────────────────────────

    def _summarize_tool_result(self, content: str) -> str:
        lines = content.splitlines()
        url, title, snippet_lines = "", "", []
        in_snapshot = False
        for line in lines:
            if "Page URL:" in line:
                url = line.split("Page URL:")[-1].strip()
            elif "Page Title:" in line:
                title = line.split("Page Title:")[-1].strip()
            elif line.strip().startswith("```yaml") or line.strip().startswith("### Snapshot"):
                in_snapshot = True
            elif in_snapshot and line.strip() and not line.startswith("```"):
                snippet_lines.append(line.strip())
                if len(snippet_lines) >= 3:
                    break
        parts = []
        if title:        parts.append(f"Page: {title}")
        if url:          parts.append(f"URL: {url}")
        if snippet_lines: parts.append("Snapshot: " + " | ".join(snippet_lines))
        return " — ".join(parts) if parts else content[:200]

    def _group_messages(self, messages: list[dict]) -> list[list[dict]]:
        """Group flat message list into logical units — each a complete pair or standalone."""
        groups = []
        i = 0
        while i < len(messages):
            m = messages[i]
            if m["role"] == "assistant" and m.get("tool_calls"):
                ids = {tc["id"] for tc in m["tool_calls"]}
                pair = [m]
                i += 1
                while i < len(messages) and messages[i]["role"] == "tool":
                    if messages[i].get("tool_call_id") in ids:
                        pair.append(messages[i])
                        ids.discard(messages[i]["tool_call_id"])
                    i += 1
                groups.append(pair)
            else:
                groups.append([m])
                i += 1
        return groups

    def _build_context(self) -> list[dict]:
        system = [m for m in self.state.messages if m["role"] == "system"]

        # Append live consideration set to system prompt so LLM always knows what's on the table
        if self.state.consideration_set or self.state.shortlist:
            lines = ["", "── CURRENT STATE ──"]
            if self.state.consideration_set:
                lines.append(f"Consideration set ({len(self.state.consideration_set)}/8 slots):")
                for i, p in enumerate(self.state.consideration_set):
                    lines.append(f"  {i+1}. {p['name']} — {p['price']} ({p['store']})")
            if self.state.shortlist:
                lines.append("Shortlist (user approved for price hunting):")
                for p in self.state.shortlist:
                    lines.append(f"  • {p['name']} — {p['store']}")
            system = [{**system[0], "content": system[0]["content"] + "\n".join(lines)}] if system else system

        rest   = [m for m in self.state.messages if m["role"] != "system"]

        groups = self._group_messages(rest)

        # Pin first user message, trim from the middle by dropping oldest pairs
        pinned    = groups[:1] if groups else []
        trimmable = groups[1:]

        def _tokens(gs):
            return _count_tokens([m for g in gs for m in g])

        while _tokens(pinned + trimmable) > TOKEN_BUDGET and len(trimmable) > 1:
            trimmable.pop(0)

        kept = pinned + trimmable

        # Snapshots: only the most recent browser_snapshot result stays full.
        # All earlier ones are summarized immediately — stale DOM trees are pure bloat.
        snapshot_groups = [
            g for g in kept
            if any(m["role"] == "tool" and m.get("_name") == "browser_snapshot" for m in g)
        ]
        stale_snapshot_ids = {id(g) for g in snapshot_groups[:-1]}

        # Other tool results: summarize beyond TOOL_RESULT_KEEP_FULL
        tool_pairs = [g for g in kept if any(m["role"] == "tool" for m in g)]
        cutoff_ids = {
            id(g) for g in tool_pairs[:-TOOL_RESULT_KEEP_FULL]
        } if len(tool_pairs) > TOOL_RESULT_KEEP_FULL else set()

        summarize_ids = stale_snapshot_ids | cutoff_ids

        trimmed = []
        for group in kept:
            if id(group) in summarize_ids:
                summarized = []
                for m in group:
                    if m["role"] == "tool":
                        tc_id = m.get("tool_call_id", "")
                        if tc_id not in self.state.summary_cache:
                            self.state.summary_cache[tc_id] = self._summarize_tool_result(m["content"] or "")
                            print(f"[Summary] {self.state.summary_cache[tc_id][:80]}")
                        m = {**m, "content": self.state.summary_cache[tc_id]}
                    summarized.append(m)
                trimmed.extend(summarized)
            else:
                trimmed.extend(group)

        # Strip private metadata fields (_name, _llm_time_s, _browser_time_s) before sending to LLM
        def _strip(m: dict) -> dict:
            return {k: v for k, v in m.items() if not k.startswith("_")}

        context = [_strip(m) for m in system + trimmed]
        STATE_LOG.write_text(json.dumps(self.state.messages, indent=2))
        CONTEXT_LOG.write_text(json.dumps(context, indent=2))
        return context

    def _heal_messages(self):
        """Remove any trailing incomplete tool call pairs from state.messages.
        Called at the start of each run() to fix state left by a previous crash."""
        msgs = self.state.messages
        while msgs:
            last = msgs[-1]
            if last["role"] == "assistant" and last.get("tool_calls"):
                expected = {tc["id"] for tc in last["tool_calls"]}
                # check how many results follow — there are none since it's the last message
                msgs.pop()
                print(f"[Heal] Removed incomplete assistant tool_calls with no results: {[tc['function']['name'] for tc in last['tool_calls']]}")
            elif last["role"] == "tool":
                # walk back to find the assistant message and check if all results are present
                tool_ids_present = set()
                i = len(msgs) - 1
                while i >= 0 and msgs[i]["role"] == "tool":
                    tool_ids_present.add(msgs[i].get("tool_call_id"))
                    i -= 1
                if i >= 0 and msgs[i]["role"] == "assistant" and msgs[i].get("tool_calls"):
                    expected = {tc["id"] for tc in msgs[i]["tool_calls"]}
                    if tool_ids_present == expected:
                        break  # pair is complete, stop healing
                    # partial results — remove all of them plus the assistant message
                    while len(msgs) > i:
                        msgs.pop()
                    print(f"[Heal] Removed partial tool call pair")
                else:
                    break
            else:
                break

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _think(self, context: list[dict]):
        """Main LLM call — returns (message, elapsed_seconds)."""
        t0       = time.time()
        response = self.llm.chat.completions.create(
            model=self.model, messages=context, tools=self.tools,
        )
        elapsed = time.time() - t0
        print(f"[Time] LLM: {elapsed:.1f}s")
        return response.choices[0].message, elapsed

    def _judge(self, task: str, answer: str, judge_extra: str = "") -> tuple[bool, str]:
        visited = "\n".join(f"- {u}" for u in self.state.visited_urls) or "- (no pages visited)"
        evidence_lines = "\n".join(
            f"- screenshot: {e['screenshot']} (at {e['url']})" for e in self.state.evidence
        ) if self.state.evidence else "- (no screenshots captured)"
        extra = f"\nMODE-SPECIFIC CRITERIA:\n{judge_extra}" if judge_extra else ""
        prompt = f"""You are a strict judge evaluating a browser agent's answer.

TASK: {task}
PAGES VISITED:\n{visited}
SCREENSHOTS TAKEN:\n{evidence_lines}
AGENT'S ANSWER:\n{answer}{extra}

Respond in JSON: {{"sufficient": true/false, "feedback": "one sentence"}}
Be strict. Vague or generic answers without specific details are NOT sufficient."""

        print("\n[Judge] Evaluating answer...")
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        try:
            result     = json.loads(response.choices[0].message.content)
            sufficient = result.get("sufficient", True)
            feedback   = result.get("feedback", "")
            print(f"[Judge] {'sufficient' if sufficient else 'NOT sufficient'} — {feedback}")
            return sufficient, feedback
        except Exception:
            return True, ""

    async def _check_screenshot(self, images: list, task: str = "", step: int = 0) -> str:
        """Send screenshot to LLM. On product listing pages, lists visible product names; otherwise one-sentence description.
        Saves the raw PNG + description txt to screenshots/ for debugging."""
        if not images:
            return "[screenshot unavailable]"

        t0 = time.time()
        if task:
            prompt = (
                f"A browser agent is working on: \"{task}\"\n\n"
                "Look at this screenshot carefully.\n\n"
                "If this is a product listing, search results, or category page showing MULTIPLE products:\n"
                "Respond with LISTING: then for each visible product (up to 8) write one line: "
                "'- [product name] | [color/colorway] | [style descriptor, e.g. low-top, chunky, minimal, retro]'\n"
                "Be specific about colors — say 'pastel lavender' not just 'purple', 'off-white' not just 'white'.\n\n"
                "If this is a single product detail page or any other page:\n"
                "Respond with PAGE: then one sentence describing what's shown."
            )
        else:
            prompt = "Describe this webpage in one sentence: what site is this and what is shown on screen?"
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                *images,
            ]}],
        )
        print(f"[Time] LLM (screenshot): {time.time() - t0:.1f}s")
        description = response.choices[0].message.content or "[could not describe screenshot]"

        # Save LLM description alongside the PNG
        try:
            txt_path = SCREENSHOTS_DIR / f"step_{step:03d}.txt"
            txt_path.write_text(description)
        except Exception as e:
            print(f"[Screenshot] Failed to save txt: {e}")

        return description

    async def _generate_title(self, task: str) -> str:
        """Quick LLM call to generate a fun contextual title for the UI header."""
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"Write a short, fun, enthusiastic title (max 8 words, no quotes, no leading punctuation) for this task: {task}"}],
            )
            return response.choices[0].message.content.strip().strip('"').strip("'")
        except Exception:
            return "Let's get started!"

    async def _fire_title(self, task: str):
        title = await self._generate_title(task)
        if self.on_event:
            await self.on_event({"type": "update_title", "text": title})

    # ── Tool execution ────────────────────────────────────────────────────────

    async def _execute_tool(self, tc, task: str) -> dict | None:
        """Execute one tool call. Returns the tool result dict (to be appended by caller),
        or None for ask_human (which appends directly and is always a single call)."""
        name = tc.function.name
        args = json.loads(tc.function.arguments)

        # Track URLs from navigate args
        if name == "browser_navigate" and "url" in args:
            self._track_url(args["url"])
        if name in ("navigate_page", "new_page") and "url" in args:
            self._track_url(args["url"])

        # ask_human — cap at 2 questions, then route to web UI or terminal
        if name == "ask_human":
            question = args.get("question", "")
            print(f"\n[Agent asks] {question}")
            if self.on_ask_human:
                human_response = await self.on_ask_human(question)
            else:
                human_response = input("Your answer: ").strip()
            print(f"[Human] {human_response}")
            return {"role": "tool", "tool_call_id": tc.id, "content": human_response, "_name": "ask_human", "_browser_time_s": 0}

        # update_consideration — flush or shortlist based on user response
        if name == "update_consideration":
            action = args.get("action", "flush")
            indices = [i - 1 for i in args.get("selected_indices", [])]  # convert to 0-based

            if action == "shortlist" and indices:
                selected = [self.state.consideration_set[i] for i in indices if i < len(self.state.consideration_set)]
                rejected = [p for i, p in enumerate(self.state.consideration_set) if i not in indices]
                self.state.shortlist.extend(selected)
                keep_urls = [p["url"] for p in selected]
                names = ", ".join(p["name"] for p in selected)
                print(f"[Consideration] Shortlisted: {names}")
                # Tell UI to remove the rejected cards
                if self.on_event:
                    await self.on_event({"type": "keep_cards", "urls": keep_urls})
                    await self.on_event({"type": "update_title", "text": "Let me find more options!"})
                self.state.consideration_set = []
                self.state.emitted_urls = set()  # allow finding fresh products
                return {"role": "tool", "tool_call_id": tc.id, "content": f"Shortlisted: {names}. Shortlist now has {len(self.state.shortlist)} item(s). Consideration set cleared. Find 4 more products across different brand sites (2 per brand minimum) to show the user — do NOT start price hunting yet unless the user explicitly asked for it.", "_name": "update_consideration", "_browser_time_s": 0}
            else:
                print(f"[Consideration] Flushed — starting new round")
                if self.on_event:
                    await self.on_event({"type": "keep_cards", "urls": []})
                    await self.on_event({"type": "update_title", "text": "Let me try something different!"})
                self.state.consideration_set = []
                self.state.emitted_urls = set()
                return {"role": "tool", "tool_call_id": tc.id, "content": "All rejected. Cards cleared. Incorporate the user's feedback and find 8 completely new options across different brands (2 per brand minimum).", "_name": "update_consideration", "_browser_time_s": 0}

        # emit_product_card — grab og:image from current page, fire callback, log to console
        if name == "emit_product_card":
            url = args.get("url", "")
            if url and url in self.state.emitted_urls:
                print(f"[Card] Duplicate skipped: {url}")
                return {"role": "tool", "tool_call_id": tc.id, "content": "Already emitted a card for this product. Navigate to a completely different store or product page.", "_name": "emit_product_card", "_browser_time_s": 0}
            if url:
                self.state.emitted_urls.add(url)
            image_url = await self.browser.get_og_image()
            card = {
                "store":         args.get("store", ""),
                "name":          args.get("name", ""),
                "price":         args.get("price", ""),
                "url":           url,
                "detail_label":  args.get("detail_label", ""),
                "detail_value":  args.get("detail_value", ""),
                "justification": args.get("justification", ""),
                "image_url":     image_url,
            }
            print(f"[Card] {card['store']} — {card['name']} — {card['price']} | {card['detail_label']}: {card['detail_value']}")
            print(f"       {card['justification']}")
            if self.on_card:
                await self.on_card(card)

            # Add to consideration set
            self.state.consideration_set.append({"name": card["name"], "store": card["store"], "price": card["price"], "url": card["url"]})
            count = len(self.state.consideration_set)
            same_brand_count = sum(1 for p in self.state.consideration_set if p["store"] == card["store"])
            print(f"[Consideration] {count}/8 slots filled | {card['store']}: {same_brand_count} product(s)")

            if count == 1 and self.on_event:
                await self.on_event({"type": "update_title", "text": "How do these look?"})

            if count < 8:
                if same_brand_count < 2:
                    return {"role": "tool", "tool_call_id": tc.id, "content": f"Card emitted ({count}/8). You only have {same_brand_count} product from {card['store']} — stay on this brand and find at least one more option here before moving to the next brand.", "_name": "emit_product_card", "_browser_time_s": 0}
                else:
                    return {"role": "tool", "tool_call_id": tc.id, "content": f"Card emitted ({count}/8). Good — {same_brand_count} products from {card['store']}. Now move to the next brand site and repeat (2+ products per brand).", "_name": "emit_product_card", "_browser_time_s": 0}
            else:
                names = ", ".join(f"{i+1}. {p['name']} ({p['price']})" for i, p in enumerate(self.state.consideration_set))
                return {"role": "tool", "tool_call_id": tc.id, "content": f"Consideration set full (8/8). Use ask_human to present all options and get feedback: {names}", "_name": "emit_product_card", "_browser_time_s": 0}

        # Log browser_run_code to file instead of console
        if name == "browser_run_code":
            code  = args.get("code", "")
            entry = f"\n{'─'*60}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] task: {task}\n{code}\n"
            PLAYWRIGHT_LOG.open("a").write(entry)
            print(f"[Tool] browser_run_code ({len(code)} chars → playwright_code.log)")
        else:
            print(f"[Tool] {name}({args})")

        # Emit thinking event for the web UI
        if self.on_thinking:
            thinking_text = {
                "browser_navigate":        f"Navigating to {args.get('url', '')}",
                "browser_snapshot":        "Reading page content",
                "browser_click":           f"Clicking: {args.get('element') or 'index ' + str(args.get('index', ''))}",
                "browser_type":            f"Typing: {args.get('text', '')}",
                "browser_scroll":          f"Scrolling {args.get('direction', 'down')}",
                "browser_press_key":       f"Pressing {args.get('key', '')}",
                "browser_take_screenshot": "Taking screenshot",
                "browser_back":            "Going back",
                "browser_forward":         "Going forward",
            }.get(name, name)
            await self.on_thinking(thinking_text)

        # Execute against browser
        t0 = time.time()
        try:
            text, images = await asyncio.wait_for(self.browser.call_tool(name, args), timeout=15.0)
        except asyncio.TimeoutError:
            text, images = "[Tool timed out after 15s]", []
        print(f"[Time] Browser ({self._browser_type}): {time.time() - t0:.1f}s")

        # For screenshots: save PNG to disk, summarise via LLM, return text only
        if name == "browser_take_screenshot" and images:
            # Save raw PNG immediately before anything else touches the image data
            try:
                import base64 as _b64
                img_data = images[0].get("image_url", {}).get("url", "")
                if img_data.startswith("data:image/png;base64,"):
                    png_bytes = _b64.b64decode(img_data.split(",", 1)[1])
                    png_path = SCREENSHOTS_DIR / f"step_{self._current_step:03d}.png"
                    png_path.write_bytes(png_bytes)
                    print(f"[Screenshot] Saved {png_path.name} ({len(png_bytes):,} bytes)")
                else:
                    print(f"[Screenshot] Unexpected format: {img_data[:60]!r}")
            except Exception as e:
                print(f"[Screenshot] Save failed: {type(e).__name__}: {e}")
            text = await self._check_screenshot(images, task=task, step=self._current_step)
            images = []

        # Truncate large snapshots
        raw_len = len(text)
        if self._max_snapshot and raw_len > self._max_snapshot:
            text = text[:self._max_snapshot] + f"\n… [truncated: {raw_len - self._max_snapshot} more chars]"

        preview = next(
            (line.strip() for line in text.splitlines() if "Page URL:" in line or "Page Title:" in line),
            text[:80]
        )
        print(f"[Tool] → {raw_len} chars | {preview}")

        # Track URLs from tool result text
        for line in text.splitlines():
            if "Page URL:" in line:
                self._track_url(line.split("Page URL:")[-1].strip())
            elif 'url="http' in line:
                url = line.split('url="')[1].split('"')[0]
                self._track_url(url)

        # Track screenshots saved to disk as evidence
        if name in ("take_screenshot", "browser_take_screenshot"):
            path = args.get("filePath", "")
            if path:
                current_url = self.state.visited_urls[-1] if self.state.visited_urls else ""
                self.state.evidence.append({"url": current_url, "screenshot": path})
                print(f"[Evidence] screenshot saved: {path}")

        return {"role": "tool", "tool_call_id": tc.id, "content": text, "_images": images, "_name": name, "_browser_time_s": round(time.time() - t0, 2)}

    # ── Core loop ─────────────────────────────────────────────────────────────

    async def run(self, task: str) -> str:
        print(f"[Agent] Task: {task}\n")

        self.state.task = task
        self.state.visited_urls = []
        self.state.evidence = []
        self.state.judge_rounds = 0
        self.state.tool_outputs = {}
        self.state.emitted_urls = set()
        self.state.consideration_set = []
        self.state.shortlist = []
        self._heal_messages()
        self.state.messages.append({"role": "user", "content": task})

        # Fire a fun contextual title for the UI — runs in background, doesn't block the loop
        if self.on_event:
            asyncio.create_task(self._fire_title(task))

        judge_extra = TASK_MODES.get(self.mode, {}).get("judge_extra", "") if self.mode else ""

        _last_snapshot_fp = ""   # fingerprint of last snapshot content
        _same_snapshot_count = 0

        # Clear screenshots dir for this run
        if SCREENSHOTS_DIR.exists():
            for f in SCREENSHOTS_DIR.glob("step_*"):
                f.unlink()
        SCREENSHOTS_DIR.mkdir(exist_ok=True)

        for step in range(MAX_STEPS):
            self._current_step = step
            self._heal_messages()
            context = self._build_context()
            print(f"[Context] {len(context)} messages | ~{_count_tokens(context):,} tokens")

            message, llm_time = self._think(context)

            # ── No tool calls: agent proposes an answer ──
            if not message.tool_calls:
                self.state.messages.append({"role": "assistant", "content": message.content, "_llm_time_s": round(llm_time, 2)})
                print(f"\n[Agent] Proposed answer after {step + 1} step(s)")

                if self.state.judge_rounds < MAX_JUDGE_ROUNDS:
                    sufficient, feedback = self._judge(task, message.content, judge_extra)
                    self.state.judge_rounds += 1
                    if not sufficient:
                        self.state.messages.append({"role": "user", "content": f"[Judge feedback — internal only, NOT from the user] {feedback} Continue researching. Do NOT call update_consideration or flush the consideration set based on this message — only the user can approve or reject products."})
                        continue

                print(f"[Agent] Done (judge rounds: {self.state.judge_rounds})")
                return message.content

            # ── Tool calls: execute all, then append atomically ──
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ],
                "_llm_time_s": round(llm_time, 2),
            }

            results = []
            for tc in message.tool_calls:
                result = await self._execute_tool(tc, task)
                if result is not None:
                    results.append(result)

            # Commit assistant message + all results together — never a partial pair
            self.state.messages.append(assistant_msg)
            for r in results:
                self.state.messages.append({
                    "role": "tool",
                    "tool_call_id": r["tool_call_id"],
                    "content": r["content"],
                    "_name": r.get("_name", ""),
                    "_browser_time_s": r.get("_browser_time_s", 0),
                })

            # Stuck detector — fires only when the exact same snapshot DOM appears repeatedly.
            # A different product page, a search results page, even a scroll produces a different
            # fingerprint and resets the counter. Only a truly frozen/blocked page repeats identically.
            snapshot_results = [r for r in results if r.get("_name") == "browser_snapshot"]
            for snap in snapshot_results:
                content = snap.get("content", "")
                fp = content[:500]          # URL + title + first handful of elements
                element_count = content.count("\n[")
                is_blank = element_count < 4

                if fp and fp == _last_snapshot_fp:
                    _same_snapshot_count += 1
                    threshold = 3 if is_blank else 8   # exit blank pages faster
                    print(f"[Stuck] Identical snapshot x{_same_snapshot_count} (blank={is_blank})")
                    if _same_snapshot_count >= threshold:
                        print(f"[Stuck] Nudging away after {_same_snapshot_count} identical snapshots")
                        self.state.messages.append({"role": "user", "content": "[System] The page hasn't changed across multiple attempts — it is likely blocked or frozen. Navigate away immediately and try a completely different website."})
                        _same_snapshot_count = 0
                        _last_snapshot_fp = ""
                else:
                    _same_snapshot_count = 0
                    _last_snapshot_fp = fp

        return "Reached max steps without completing the task."
