import json
import os
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from mcp_client import BrowserMCPClient  # wraps both @playwright/mcp and chrome-devtools-mcp
from direct_browser_client import DirectBrowserClient

# ── Config ────────────────────────────────────────────────────────────────────

BACKENDS = {
    "ollama": {
        "base_url":     "http://localhost:11434/v1",
        "api_key":      "ollama",
        "model":        "gemma4:31b-cloud",
        "max_snapshot": 6_000,
    },
    "openai": {
        "base_url":     None,
        "api_key":      os.getenv("OPENAI_API_KEY", ""),
        "model":        "gpt-4o",
        "max_snapshot": None,
    },
}

MAX_STEPS        = 100
MAX_JUDGE_ROUNDS = 3
PLAYWRIGHT_LOG   = Path(__file__).parent / "playwright_code.log"

# Context window settings
TOOL_RESULT_MAX       = 4000
TOOL_RESULT_KEEP_FULL = 6
TOOL_RESULT_TRIM_TO   = 300
ROLLING_WINDOW        = 40

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
- When writing browser_run_code, write Python async Playwright code. `page` and `context` are available. Use `await` for all calls."""

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
1. Search for the item on Google or a shopping site.
2. From the search/listing page, pick 3-5 specific products that look promising.
3. For EACH product: open a NEW TAB first, then navigate to the product URL in that tab. Never open products in the same tab.
4. On each product page: scroll down to see full details, price, and reviews. Take a screenshot.
5. After visiting ALL product pages, write your final answer with: product name, price, rating, key specs, and the URL where you saw it.
- A search results page or listing grid is NOT a product page. You must click through.
- Do not give a final answer until you have visited at least 3 individual product pages.""",
        "judge_extra": "STRICT CHECK: The agent must have opened at least 3 individual product pages IN SEPARATE TABS (not google.com, not search result pages, not category listings). Each product page must be a distinct URL on a retailer site. If the visited URLs are all google.com or search pages, the answer is NOT sufficient.",
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_tokens(messages: list[dict]) -> int:
    return len(json.dumps(messages)) // 4


# ── Agent ─────────────────────────────────────────────────────────────────────

class BrowserAgent:
    def __init__(self, backend: str | None = None, model: str | None = None, browser: str = "direct"):
        b = backend or os.getenv("AGENT_BACKEND", "ollama")
        cfg = BACKENDS[b]
        self.model         = model or cfg["model"]
        self._max_snapshot = cfg.get("max_snapshot")
        self._browser_type = browser

        self.llm = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

        if browser == "direct":
            self.browser = DirectBrowserClient()
        elif browser == "cdp":
            self.browser = BrowserMCPClient(server="cdp")
        else:
            self.browser = BrowserMCPClient(server="playwright")

        self.tools: list[dict]       = []
        self.messages: list[dict]    = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._summary_cache: dict     = {}
        self._visited_urls: list[str]  = []
        self._evidence: list[dict]     = []  # {url, screenshot_path, note}
        self._current_mode: str | None = None

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
        self._current_mode = mode
        mode_cfg       = TASK_MODES.get(mode, {}) if mode else {}
        system_content = SYSTEM_PROMPT + mode_cfg.get("prompt", "")
        self.messages  = [m for m in self.messages if m["role"] != "system"]
        self.messages.insert(0, {"role": "system", "content": system_content})
        print(f"[Mode] {mode or 'none'}")

    def is_continuation(self, task: str) -> bool:
        non_system = [m for m in self.messages if m["role"] != "system"]
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
        if not self._visited_urls or self._visited_urls[-1] != url:
            self._visited_urls.append(url)
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

    def _build_context(self) -> list[dict]:
        system = [m for m in self.messages if m["role"] == "system"]
        rest   = [m for m in self.messages if m["role"] != "system"]
        rest   = rest[-ROLLING_WINDOW:]

        tool_indices = [i for i, m in enumerate(rest) if m["role"] == "tool"]
        cutoff = tool_indices[-TOOL_RESULT_KEEP_FULL] if len(tool_indices) > TOOL_RESULT_KEEP_FULL else 0

        trimmed = []
        for i, m in enumerate(rest):
            if i < cutoff and m["role"] == "tool":
                tc_id = m.get("tool_call_id", "")
                if tc_id not in self._summary_cache:
                    self._summary_cache[tc_id] = self._summarize_tool_result(m["content"] or "")
                    print(f"[Summary] {self._summary_cache[tc_id]}")
                m = {**m, "content": self._summary_cache[tc_id]}
            trimmed.append(m)

        valid_ids = {tc["id"] for m in trimmed if m["role"] == "assistant" for tc in m.get("tool_calls", [])}
        trimmed   = [m for m in trimmed if not (m["role"] == "tool" and m.get("tool_call_id") not in valid_ids)]

        return system + trimmed

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
        visited = "\n".join(f"- {u}" for u in self._visited_urls) or "- (no pages visited)"
        evidence_lines = "\n".join(
            f"- screenshot: {e['screenshot']} (at {e['url']})" for e in self._evidence
        ) if self._evidence else "- (no screenshots captured)"
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
        context = self._build_context()
        context.append({"role": "user", "content": [
            {"type": "text", "text": "Screenshot taken after last action. Confirm what page you are on and whether it looks correct. Be brief."},
            *images,
        ]})
        t0       = time.time()
        response = self.llm.chat.completions.create(model=self.model, messages=context)
        print(f"[Time] LLM (screenshot check): {time.time() - t0:.1f}s")
        confirmation = response.choices[0].message.content or f"[confirmed after {trigger}]"
        print(f"[Screenshot] {confirmation[:120]}")
        self.messages.append({"role": "user", "content": f"[Visual check after {trigger}]: {confirmation}"})

    # ── Tool execution ────────────────────────────────────────────────────────

    async def _execute_tool(self, tc, task: str) -> None:
        name = tc.function.name
        args = json.loads(tc.function.arguments)

        # Track URLs from navigate args (Playwright MCP)
        if name == "browser_navigate" and "url" in args:
            self._track_url(args["url"])

        # Track URLs from CDP navigate/new_page args
        if name in ("navigate_page", "new_page") and "url" in args:
            self._track_url(args["url"])

        # ask_human is handled locally, not sent to browser
        if name == "ask_human":
            print(f"\n[Agent asks] {args.get('question', '')}")
            human_response = input("Your answer: ").strip()
            print()
            self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": human_response})
            return

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

        # Track URLs from tool result text — handles both Playwright (Page URL:) and CDP (url="...")
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
                current_url = self._visited_urls[-1] if self._visited_urls else ""
                self._evidence.append({"url": current_url, "screenshot": path})
                print(f"[Evidence] screenshot saved: {path}")

        self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})

        # Auto-screenshot after navigation/clicks
        if images or name in ("browser_navigate", "browser_click"):
            await self._check_screenshot(name, images)

    # ── Core loop ─────────────────────────────────────────────────────────────

    async def run(self, task: str) -> str:
        print(f"[Agent] Task: {task}\n")
        self.messages.append({"role": "user", "content": task})
        self._visited_urls = []
        self._evidence     = []
        judge_extra  = TASK_MODES.get(self._current_mode, {}).get("judge_extra", "") if self._current_mode else ""
        judge_rounds = 0

        for step in range(MAX_STEPS):
            context = self._build_context()
            print(f"[Context] {len(context)} messages | ~{_count_tokens(context):,} tokens")

            message = self._think(context)

            # ── No tool calls: agent proposes an answer ──
            if not message.tool_calls:
                self.messages.append({"role": "assistant", "content": message.content})
                print(f"\n[Agent] Proposed answer after {step + 1} step(s)")

                if judge_rounds < MAX_JUDGE_ROUNDS:
                    sufficient, feedback = self._judge(task, message.content, judge_extra)
                    judge_rounds += 1
                    if not sufficient:
                        self.messages.append({"role": "user", "content": f"[Judge feedback] {feedback} Please continue researching."})
                        continue

                print(f"[Agent] Done (judge rounds: {judge_rounds})")
                return message.content

            # ── Tool calls: execute each and loop ──
            self.messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ],
            })

            for tc in message.tool_calls:
                await self._execute_tool(tc, task)

        return "Reached max steps without completing the task."
