"""
direct_browser_client.py

Drop-in replacement for PlaywrightMCPClient.
Uses Playwright directly with a persistent browser profile so cookies and
session data survive between runs — same behaviour as the MCP server.

Same interface: connect(), get_tools(), call_tool(), close()
"""
import asyncio
import base64
from pathlib import Path
from playwright.async_api import async_playwright, BrowserContext, Page

PROFILE_DIR = Path(__file__).parent / ".browser_profile"

TOOLS = [
    {"type": "function", "function": {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Full URL to navigate to"}
        }, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "browser_snapshot",
        "description": "Capture an accessibility tree snapshot of the current page. Use this to read page content and find element refs for clicking/typing.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "browser_take_screenshot",
        "description": "Take a screenshot of the current page.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "browser_click",
        "description": "Click an element on the page identified by its ref from the snapshot.",
        "parameters": {"type": "object", "properties": {
            "element": {"type": "string", "description": "Human-readable element description (for logging)"},
            "ref":     {"type": "string", "description": "Element ref from the snapshot (e.g. e42)"},
        }, "required": ["element", "ref"]},
    }},
    {"type": "function", "function": {
        "name": "browser_type",
        "description": "Type text into an input field identified by its ref.",
        "parameters": {"type": "object", "properties": {
            "ref":  {"type": "string", "description": "Element ref from the snapshot"},
            "text": {"type": "string", "description": "Text to type"},
        }, "required": ["ref", "text"]},
    }},
    {"type": "function", "function": {
        "name": "browser_press_key",
        "description": "Press a keyboard key (e.g. Enter, Tab, Escape, ArrowDown).",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "Key name"}
        }, "required": ["key"]},
    }},
    {"type": "function", "function": {
        "name": "browser_scroll",
        "description": "Scroll the page up or down.",
        "parameters": {"type": "object", "properties": {
            "direction":  {"type": "string", "enum": ["up", "down", "left", "right"]},
            "coordinate": {"type": "array", "items": {"type": "number"},
                           "description": "[x, y] pixel coordinate to scroll at (defaults to centre)"},
        }, "required": ["direction"]},
    }},
    {"type": "function", "function": {
        "name": "browser_tabs",
        "description": "Manage browser tabs.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["new", "list", "switch", "close"],
                       "description": "new=open tab, list=list tabs, switch=focus tab, close=close current tab"},
            "url":   {"type": "string", "description": "URL to open (for action=new)"},
            "index": {"type": "integer", "description": "Tab index to switch to (for action=switch)"},
        }, "required": ["action"]},
    }},
    {"type": "function", "function": {
        "name": "browser_run_code",
        "description": (
            "Run Python async Playwright code directly. "
            "`page` (current Page) and `context` (BrowserContext) are available as locals. "
            "Use `await` for all async calls. Example:\n"
            "  await page.get_by_role('button', name='Add to cart').click()\n"
            "  text = await page.locator('.price').inner_text()"
        ),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python async Playwright code to run"}
        }, "required": ["code"]},
    }},
    {"type": "function", "function": {
        "name": "browser_back",
        "description": "Navigate back in browser history.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "browser_forward",
        "description": "Navigate forward in browser history.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


class DirectBrowserClient:
    def __init__(self, profile_dir: Path = PROFILE_DIR):
        self.profile_dir = profile_dir
        self._pw = None
        self._context: BrowserContext | None = None
        self._pages: list[Page] = []
        self._page_idx: int = 0

    @property
    def _page(self) -> Page:
        return self._pages[self._page_idx]

    # ------------------------------------------------------------------
    # Public interface (matches PlaywrightMCPClient)
    # ------------------------------------------------------------------

    async def connect(self):
        self._pw = await async_playwright().start()
        self._context = await self._pw.chromium.launch_persistent_context(
            str(self.profile_dir),
            headless=False,
        )
        if self._context.pages:
            self._pages = list(self._context.pages)
        else:
            self._pages = [await self._context.new_page()]
        print(f"[Browser] Connected — persistent profile at {self.profile_dir.name}/")

    async def get_tools(self) -> list[dict]:
        return TOOLS

    async def call_tool(self, name: str, args: dict) -> tuple[str, list[dict]]:
        handlers = {
            "browser_navigate":       self._navigate,
            "browser_snapshot":       self._snapshot,
            "browser_take_screenshot": self._screenshot,
            "browser_click":          self._click,
            "browser_type":           self._type,
            "browser_press_key":      self._press_key,
            "browser_scroll":         self._scroll,
            "browser_tabs":           self._tabs,
            "browser_run_code":       self._run_code,
            "browser_back":           self._back,
            "browser_forward":        self._forward,
        }
        handler = handlers.get(name)
        if not handler:
            return f"[Tool not implemented: {name}]", []
        return await handler(args)

    async def close(self):
        if self._context:
            await self._context.close()
        if self._pw:
            await self._pw.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _page_state(self) -> str:
        url = self._page.url
        try:
            title = await self._page.title()
        except Exception:
            title = ""
        return f"- Page URL: {url}\n- Page Title: {title}"

    async def _resolve_ref(self, ref: str, element: str = "") -> bool:
        """Try to click by ref, fall back to text. Returns True on success."""
        # Try aria-ref locator (Playwright 1.49+)
        if ref:
            try:
                await self._page.locator(f"aria-ref={ref}").click(timeout=4000)
                return True
            except Exception:
                pass
        # Fall back to element description text
        if element:
            try:
                await self._page.get_by_text(element, exact=False).first.click(timeout=4000)
                return True
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _navigate(self, args: dict) -> tuple[str, list]:
        await self._page.goto(args["url"], wait_until="domcontentloaded")
        state = await self._page_state()
        return state, []

    async def _snapshot(self, args: dict) -> tuple[str, list]:
        # Request refs so the LLM can reference elements in follow-up calls
        snap = await self._page.locator("body").aria_snapshot()
        state = await self._page_state()
        return f"{state}\n\n{snap}", []

    async def _screenshot(self, args: dict) -> tuple[str, list]:
        data = await self._page.screenshot()
        b64 = base64.b64encode(data).decode()
        images = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        state = await self._page_state()
        return state, images

    async def _click(self, args: dict) -> tuple[str, list]:
        ref = args.get("ref", "")
        element = args.get("element", "")
        ok = await self._resolve_ref(ref, element)
        state = await self._page_state()
        if not ok:
            return f"[Click failed — could not locate '{element}' (ref={ref})]\n{state}", []
        return state, []

    async def _type(self, args: dict) -> tuple[str, list]:
        ref = args.get("ref", "")
        text = args.get("text", "")
        try:
            await self._page.locator(f"aria-ref={ref}").fill(text, timeout=4000)
        except Exception:
            # Fall back: click to focus then keyboard type
            try:
                await self._resolve_ref(ref)
                await self._page.keyboard.type(text)
            except Exception as e:
                return f"[Type failed: {e}]", []
        state = await self._page_state()
        return state, []

    async def _press_key(self, args: dict) -> tuple[str, list]:
        await self._page.keyboard.press(args.get("key", ""))
        await self._page.wait_for_timeout(400)
        state = await self._page_state()
        return state, []

    async def _scroll(self, args: dict) -> tuple[str, list]:
        direction = args.get("direction", "down")
        coord = args.get("coordinate", None)
        delta_y = 400 if direction == "down" else (-400 if direction == "up" else 0)
        delta_x = 400 if direction == "right" else (-400 if direction == "left" else 0)
        if coord and len(coord) == 2:
            await self._page.mouse.move(coord[0], coord[1])
        await self._page.mouse.wheel(delta_x, delta_y)
        await self._page.wait_for_timeout(300)
        state = await self._page_state()
        return state, []

    async def _tabs(self, args: dict) -> tuple[str, list]:
        action = args.get("action", "list")

        if action == "new":
            new_page = await self._context.new_page()
            self._pages.append(new_page)
            self._page_idx = len(self._pages) - 1
            url = args.get("url")
            if url:
                await new_page.goto(url, wait_until="domcontentloaded")
            state = await self._page_state()
            return f"Opened new tab (index {self._page_idx}).\n{state}", []

        elif action == "list":
            lines = [
                f"Tab {i}: {p.url}{' (active)' if i == self._page_idx else ''}"
                for i, p in enumerate(self._pages)
            ]
            return "\n".join(lines), []

        elif action == "switch":
            idx = args.get("index", 0)
            if 0 <= idx < len(self._pages):
                self._page_idx = idx
                await self._pages[idx].bring_to_front()
                state = await self._page_state()
                return f"Switched to tab {idx}.\n{state}", []
            return f"[Invalid tab index: {idx}]", []

        elif action == "close":
            if len(self._pages) > 1:
                page = self._pages.pop(self._page_idx)
                await page.close()
                self._page_idx = min(self._page_idx, len(self._pages) - 1)
                state = await self._page_state()
                return f"Closed tab.\n{state}", []
            return "[Cannot close the last remaining tab]", []

        return f"[Unknown tab action: {action}]", []

    async def _run_code(self, args: dict) -> tuple[str, list]:
        """Execute Python async Playwright code with `page` and `context` in scope."""
        code = args.get("code", "")
        page = self._page
        ctx = self._context

        # Wrap in an async function so `await` works inside exec()
        indented = "\n".join(f"    {line}" for line in code.splitlines())
        wrapped = f"async def __run__(page, context):\n{indented}\n"

        exec_globals: dict = {}
        try:
            exec(wrapped, exec_globals)
            await exec_globals["__run__"](page, ctx)
            state = await self._page_state()
            return f"### Ran Playwright code\n```python\n{code}\n```\n{state}", []
        except Exception as e:
            return f"[Code error: {e}]\n```python\n{code}\n```", []

    async def _back(self, args: dict) -> tuple[str, list]:
        await self._page.go_back()
        state = await self._page_state()
        return state, []

    async def _forward(self, args: dict) -> tuple[str, list]:
        await self._page.go_forward()
        state = await self._page_state()
        return state, []
