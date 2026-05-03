import json
import os
import time
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from mcp_client import BrowserMCPClient  # wraps both @playwright/mcp and chrome-devtools-mcp
from direct_browser_client import DirectBrowserClient
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
        "model":        "gpt-4o",
        "max_snapshot": 16_000,
    },
}

MAX_STEPS        = 100
MAX_JUDGE_ROUNDS = 3
PLAYWRIGHT_LOG   = Path(__file__).parent / "playwright_code.log"
STATE_LOG        = Path(__file__).parent / "state_messages.json"
CONTEXT_LOG      = Path(__file__).parent / "context_window.json"

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
- Never navigate to a brand homepage (e.g. stevemadden.com, nike.com). Stay on aggregator/category pages like Zappos or Amazon where products are directly listed and clickable.
- If a popup or modal appears, close it first before doing anything else — look for a close/dismiss button in the snapshot and click it.
- Never construct or guess a URL from memory. Only navigate to URLs that are visible in the current page snapshot or search results."""

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

# ── Modes ─────────────────────────────────────────────────────────────────────

TASK_MODES: dict[str, dict] = {
    "shopping": {
        "description": "User wants to find, compare, or buy products - prices, deals, recommendations.",
        "prompt": """
SHOPPING MODE — follow these steps exactly:

STEP 0 — before doing anything, decide what columns to track based on the goal:
- Price comparison goal → columns: Store | Product | Price | Availability
- Style/aesthetic goal → columns: Store | Product | Description | Style notes | Price
- Feature comparison goal → columns: Store | Product | Key specs | Price | Rating
Output the empty table with headers immediately so you know what to fill in.

STEP 1 — Search for the item. Use Google Shopping or go directly to Amazon/Zappos/eBay.
STEP 2 — From the search results, click a product link directly (do not construct URLs from memory).
STEP 3 — On the product page: scroll down to see full price, availability, and details.
STEP 4 — Add a row to your comparison table with what you found on this store.
STEP 5 — Navigate back and repeat for the next store. Visit at least 3 different stores.
STEP 6 — After 3+ stores, use the completed table to write your final recommendation.

Rules:
- Never construct or guess a URL. Only navigate to URLs visible in the current snapshot or search results.
- Never open new tabs. Visit stores one at a time in the same tab, use browser_back to return.
- A search results page is NOT a product page — click through to the actual item.
- Update the comparison table after EVERY store visit, not at the end.""",
        "judge_extra": "STRICT CHECK: The agent must have visited at least 3 individual product pages on distinct retailer sites (not google.com, not search result pages). The final answer must include a comparison table with real prices/details found by actually visiting each page.",
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages)) // 4


# ── Agent ─────────────────────────────────────────────────────────────────────

class BrowserAgent:
    def __init__(self, backend: str | None = None, model: str | None = None, browser: str = "direct", visible_mouse: bool = False):
        b = backend or os.getenv("AGENT_BACKEND", "ollama")
        cfg = BACKENDS[b]
        self.model         = model or cfg["model"]
        self._max_snapshot = cfg.get("max_snapshot")
        self._browser_type = browser

        self.llm = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

        if browser == "direct":
            self.browser = DirectBrowserClient()
        elif browser == "cdp":
            self.browser = CDPBrowserClient(visible_mouse=visible_mouse)
        elif browser == "cdp-mcp":
            self.browser = BrowserMCPClient(server="cdp")
        else:
            self.browser = BrowserMCPClient(server="playwright")

        self.tools: list[dict] = []
        self.mode: str | None = None
        self.state = AgentState(messages=[{"role": "system", "content": SYSTEM_PROMPT}])

        print(f"[Agent] Backend: {b} | Model: {self.model} | Browser: {browser}")

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self):
        await self.browser.connect()
        browser_tools  = await self.browser.get_tools()
        self.tools     = browser_tools + [ASK_HUMAN_TOOL]
        print(f"[Agent] Ready — {len(browser_tools)} browser tools + ask_human\n")

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
        system   = [m for m in self.state.messages if m["role"] == "system"]
        rest     = [m for m in self.state.messages if m["role"] != "system"]

        groups   = self._group_messages(rest)

        # Pin first user message, trim from the middle by dropping oldest pairs
        pinned    = groups[:1] if groups else []
        trimmable = groups[1:]

        def _tokens(gs):
            return _count_tokens([m for g in gs for m in g])

        while _tokens(pinned + trimmable) > TOKEN_BUDGET and len(trimmable) > 1:
            trimmable.pop(0)

        kept = pinned + trimmable

        # Summarize tool results in older pairs, keep last TOOL_RESULT_KEEP_FULL pairs full
        tool_pairs = [g for g in kept if any(m["role"] == "tool" for m in g)]
        cutoff_pairs = set(
            id(g) for g in tool_pairs[:-TOOL_RESULT_KEEP_FULL]
        ) if len(tool_pairs) > TOOL_RESULT_KEEP_FULL else set()

        trimmed = []
        for group in kept:
            if id(group) in cutoff_pairs:
                summarized = []
                for m in group:
                    if m["role"] == "tool":
                        tc_id = m.get("tool_call_id", "")
                        if tc_id not in self.state.summary_cache:
                            self.state.summary_cache[tc_id] = self._summarize_tool_result(m["content"] or "")
                            print(f"[Summary] {self.state.summary_cache[tc_id]}")
                        m = {**m, "content": self.state.summary_cache[tc_id]}
                    summarized.append(m)
                trimmed.extend(summarized)
            else:
                trimmed.extend(group)

        context = system + trimmed
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
        """Main LLM call — returns the raw message."""
        t0       = time.time()
        response = self.llm.chat.completions.create(
            model=self.model, messages=context, tools=self.tools,
        )
        print(f"[Time] LLM: {time.time() - t0:.1f}s")
        return response.choices[0].message

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

    async def _check_screenshot(self, trigger: str, images: list):
        """Send screenshot to LLM for visual confirmation, store only the text reply."""
        if not images:
            try:
                _, images = await asyncio.wait_for(
                    self.browser.call_tool("browser_take_screenshot", {}), timeout=10.0
                )
            except asyncio.TimeoutError:
                print("[Screenshot] timed out — skipping")
                return
        if not images:
            return

        print(f"[Screenshot] auto after {trigger}")
        self._heal_messages()
        context = self._build_context()
        context.append({"role": "user", "content": [
            {"type": "text", "text": "Screenshot taken after last action. Reply in one sentence describing what page you are on. Do not call any tools. Plain text only."},
            *images,
        ]})
        t0 = time.time()
        # No tools passed — prevents model from outputting tool call syntax
        response = self.llm.chat.completions.create(model=self.model, messages=context)
        print(f"[Time] LLM (screenshot check): {time.time() - t0:.1f}s")
        confirmation = response.choices[0].message.content or ""
        # If model still output tool call syntax, replace with a neutral placeholder
        if not confirmation or confirmation.strip().startswith("call:") or "tool_call" in confirmation:
            confirmation = f"[page confirmed after {trigger}]"
        print(f"[Screenshot] {confirmation[:120]}")
        self.state.messages.append({"role": "user", "content": f"[Visual check after {trigger}]: {confirmation}"})

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

        # ask_human is handled locally — always a single call, safe to append directly
        if name == "ask_human":
            print(f"\n[Agent asks] {args.get('question', '')}")
            human_response = input("Your answer: ").strip()
            print()
            self.state.messages.append({"role": "tool", "tool_call_id": tc.id, "content": human_response})
            return None

        # Log browser_run_code to file instead of console
        if name == "browser_run_code":
            code  = args.get("code", "")
            entry = f"\n{'─'*60}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] task: {task}\n{code}\n"
            PLAYWRIGHT_LOG.open("a").write(entry)
            print(f"[Tool] browser_run_code ({len(code)} chars → playwright_code.log)")
        else:
            print(f"[Tool] {name}({args})")

        # Execute against browser
        t0 = time.time()
        try:
            text, images = await asyncio.wait_for(self.browser.call_tool(name, args), timeout=15.0)
        except asyncio.TimeoutError:
            text, images = "[Tool timed out after 15s]", []
        print(f"[Time] Browser ({self._browser_type}): {time.time() - t0:.1f}s")

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

        return {"role": "tool", "tool_call_id": tc.id, "content": text, "_images": images, "_name": name}

    # ── Core loop ─────────────────────────────────────────────────────────────

    async def run(self, task: str) -> str:
        print(f"[Agent] Task: {task}\n")

        self.state.task = task
        self.state.visited_urls = []
        self.state.evidence = []
        self.state.judge_rounds = 0
        self.state.tool_outputs = {}
        self._heal_messages()
        self.state.messages.append({"role": "user", "content": task})

        judge_extra = TASK_MODES.get(self.mode, {}).get("judge_extra", "") if self.mode else ""

        for step in range(MAX_STEPS):
            self._heal_messages()
            context = self._build_context()
            print(f"[Context] {len(context)} messages | ~{_count_tokens(context):,} tokens")

            message = self._think(context)

            # ── No tool calls: agent proposes an answer ──
            if not message.tool_calls:
                self.state.messages.append({"role": "assistant", "content": message.content})
                print(f"\n[Agent] Proposed answer after {step + 1} step(s)")

                if self.state.judge_rounds < MAX_JUDGE_ROUNDS:
                    sufficient, feedback = self._judge(task, message.content, judge_extra)
                    self.state.judge_rounds += 1
                    if not sufficient:
                        self.state.messages.append({"role": "user", "content": f"[Judge feedback] {feedback} Please continue researching."})
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
            }

            results = []
            for tc in message.tool_calls:
                result = await self._execute_tool(tc, task)
                if result is not None:
                    results.append(result)

            # Commit assistant message + all results together — never a partial pair
            self.state.messages.append(assistant_msg)
            for r in results:
                self.state.messages.append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]})

            # Run screenshot checks after the pair is committed
            for r in results:
                if r["_images"] or r["_name"] in ("browser_navigate", "browser_click"):
                    await self._check_screenshot(r["_name"], r["_images"])

        return "Reached max steps without completing the task."
