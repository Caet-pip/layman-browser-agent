"""
cdp_browser_client.py

Browser client using raw CDP for snapshot extraction and interaction.
Uses Playwright only to launch the browser — all page interaction goes
through CDP directly.

Same interface as DirectBrowserClient: connect(), get_tools(), call_tool(), close()
"""
import asyncio
import base64
from pathlib import Path
from playwright.async_api import async_playwright, BrowserContext, Page, CDPSession

PROFILE_DIR = Path(__file__).parent / ".browser_profile"

INTERACTIVE_ROLES = {
    "link", "button", "textbox", "searchbox", "combobox",
    "checkbox", "radio", "menuitem", "tab", "listbox",
    "option", "spinbutton", "slider", "switch", "treeitem",
}

TOOLS = [
    {"type": "function", "function": {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}
        }, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "browser_snapshot",
        "description": (
            "Capture an accessibility snapshot of the current page. "
            "Returns a numbered list of interactive elements: [1] button \"Add to Cart\". "
            "Use the number index to click or type into elements."
        ),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "browser_take_screenshot",
        "description": "Take a screenshot of the current page.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "browser_click",
        "description": "Click an element by its index number from the last snapshot.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer", "description": "Element index from the snapshot e.g. 42"},
            "element": {"type": "string", "description": "Human-readable description for logging"},
        }, "required": ["index"]},
    }},
    {"type": "function", "function": {
        "name": "browser_type",
        "description": "Type text into an input field by its index number from the last snapshot.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer", "description": "Element index from the snapshot"},
            "text": {"type": "string", "description": "Text to type"},
        }, "required": ["index", "text"]},
    }},
    {"type": "function", "function": {
        "name": "browser_press_key",
        "description": "Press a keyboard key (e.g. Enter, Tab, Escape, ArrowDown).",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string"}
        }, "required": ["key"]},
    }},
    {"type": "function", "function": {
        "name": "browser_scroll",
        "description": "Scroll the page up or down.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
        }, "required": ["direction"]},
    }},
    {"type": "function", "function": {
        "name": "browser_tabs",
        "description": "Manage browser tabs.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["new", "list", "switch", "close"]},
            "url":   {"type": "string"},
            "index": {"type": "integer"},
        }, "required": ["action"]},
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


# ── DOM extraction ─────────────────────────────────────────────────────────────

def _get_prop(node: dict, name: str) -> str:
    for prop in node.get("properties", []):
        if prop.get("name") == name:
            val = prop.get("value", {})
            return str(val.get("value", ""))
    return ""

def _get_name(node: dict) -> str:
    name_obj = node.get("name", {})
    if isinstance(name_obj, dict):
        return name_obj.get("value", "")
    return str(name_obj) if name_obj else ""

def _get_role(node: dict) -> str:
    role_obj = node.get("role", {})
    if isinstance(role_obj, dict):
        return role_obj.get("value", "")
    return str(role_obj) if role_obj else ""

def _serialize_ax_tree(nodes: list[dict]) -> tuple[str, dict[int, str]]:
    """
    Walk the AX tree, emit one line per interactive element.
    Returns (serialized_text, selector_map) where selector_map maps
    index -> nodeId for CDP interaction.
    """
    lookup = {n["nodeId"]: n for n in nodes}

    children_map: dict[str, list[str]] = {n["nodeId"]: [] for n in nodes}
    for node in nodes:
        for child_id in node.get("childIds", []):
            if child_id in children_map:
                children_map[node["nodeId"]].append(child_id)

    all_child_ids = {cid for ids in children_map.values() for cid in ids}
    roots = [n["nodeId"] for n in nodes if n["nodeId"] not in all_child_ids]
    root_id = roots[0] if roots else nodes[0]["nodeId"]

    lines = []
    counter = [1]
    selector_map: dict[int, str] = {}  # index -> nodeId

    def walk(node_id: str, depth: int = 0):
        node = lookup.get(node_id)
        if not node:
            return

        role = _get_role(node)
        name = _get_name(node)
        is_interactive = role.lower() in INTERACTIVE_ROLES

        if is_interactive:
            idx = counter[0]
            counter[0] += 1
            selector_map[idx] = node.get("backendDOMNodeId")

            parts = [f"[{idx}]", role]
            if name:
                parts.append(f'"{name}"')

            placeholder = _get_prop(node, "placeholder")
            if placeholder:
                parts.append(f'placeholder="{placeholder}"')

            value = _get_prop(node, "value")
            if value:
                parts.append(f'value="{value}"')

            checked = _get_prop(node, "checked")
            if checked and checked != "false":
                parts.append(f"checked={checked}")

            lines.append("  " * depth + " ".join(parts))

        for child_id in children_map.get(node_id, []):
            walk(child_id, depth + (1 if is_interactive else 0))

    walk(root_id)
    return "\n".join(lines), selector_map


# ── Client ─────────────────────────────────────────────────────────────────────

class CDPBrowserClient:
    def __init__(self, profile_dir: Path = PROFILE_DIR, visible_mouse: bool = False):
        self.profile_dir = profile_dir
        self.visible_mouse = visible_mouse  # True → page.mouse (visible cursor); False → raw CDP events
        self._pw = None
        self._context: BrowserContext | None = None
        self._pages: list[Page] = []
        self._page_idx: int = 0
        self._cdp: CDPSession | None = None
        self._selector_map: dict[int, str] = {}  # index -> nodeId from last snapshot

    @property
    def _page(self) -> Page:
        return self._pages[self._page_idx]

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
        await self._attach_cdp()
        for page in self._pages:
            page.on("load", lambda: asyncio.ensure_future(self._on_page_load()))
        print(f"[Browser] CDP client connected — profile: {self.profile_dir.name}/")

    async def _on_page_load(self):
        """Re-inject the cursor after every page load so it survives navigation."""
        if self.visible_mouse:
            try:
                await self._ensure_cursor()
            except Exception:
                pass

    async def _attach_cdp(self):
        if self._cdp:
            try:
                await self._cdp.detach()
            except Exception:
                pass
        self._cdp = await self._context.new_cdp_session(self._page)

    async def get_tools(self) -> list[dict]:
        return TOOLS

    async def call_tool(self, name: str, args: dict) -> tuple[str, list[dict]]:
        handlers = {
            "browser_navigate":        self._navigate,
            "browser_snapshot":        self._snapshot,
            "browser_take_screenshot": self._screenshot,
            "browser_click":           self._click,
            "browser_type":            self._type,
            "browser_press_key":       self._press_key,
            "browser_scroll":          self._scroll,
            "browser_tabs":            self._tabs,
            "browser_back":            self._back,
            "browser_forward":         self._forward,
        }
        handler = handlers.get(name)
        if not handler:
            return f"[Tool not implemented: {name}]", []
        return await handler(args)

    async def get_og_image(self) -> str:
        """Return the og:image URL from the current page, or empty string if not found."""
        try:
            return await self._page.evaluate(
                "document.querySelector('meta[property=\"og:image\"]')?.content || ''"
            )
        except Exception:
            return ""

    async def close(self):
        if self._cdp:
            try:
                await self._cdp.detach()
            except Exception:
                pass
        if self._context:
            await self._context.close()
        if self._pw:
            await self._pw.stop()

    # ── State ──────────────────────────────────────────────────────────────────

    async def _page_state(self) -> str:
        url = self._page.url
        try:
            title = await self._page.title()
        except Exception:
            title = ""
        return f"- Page URL: {url}\n- Page Title: {title}"

    # ── Tools ──────────────────────────────────────────────────────────────────

    async def _navigate(self, args: dict) -> tuple[str, list]:
        await self._page.goto(args["url"], wait_until="domcontentloaded")
        await self._page.wait_for_timeout(1500)  # let JS render after domcontentloaded
        await self._attach_cdp()
        if self.visible_mouse:
            try:
                await self._ensure_cursor()
            except Exception:
                pass  # page may still be loading — cursor will appear on next interaction
        state = await self._page_state()
        return state, []

    async def _snapshot(self, args: dict) -> tuple[str, list]:
        result = await self._cdp.send("Accessibility.getFullAXTree", {})
        nodes = result.get("nodes", [])
        serialized, selector_map = _serialize_ax_tree(nodes)
        self._selector_map = selector_map
        state = await self._page_state()
        return f"{state}\n\n{serialized}", []

    async def _screenshot(self, args: dict) -> tuple[str, list]:
        data = await self._page.screenshot()
        b64 = base64.b64encode(data).decode()
        images = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        state = await self._page_state()
        return state, images

    async def _get_element_center(self, index: int) -> tuple[float, float]:
        """Scroll element into view, then return viewport-relative center coordinates."""
        backend_node_id = self._selector_map[index]
        node_obj = await self._cdp.send("DOM.resolveNode", {"backendNodeId": int(backend_node_id)})
        obj_id = node_obj["object"]["objectId"]
        result = await self._cdp.send("Runtime.callFunctionOn", {
            "objectId": obj_id,
            "functionDeclaration": """function() {
                this.scrollIntoView({block: 'center', inline: 'center', behavior: 'instant'});
                const r = this.getBoundingClientRect();
                return {x: r.left + r.width / 2, y: r.top + r.height / 2};
            }""",
            "returnByValue": True,
        })
        coords = result["result"]["value"]
        return coords["x"], coords["y"]

    async def _ensure_cursor(self):
        """Inject a custom SVG arrow cursor into the page if not already present."""
        await self._page.evaluate("""() => {
            if (!document.body) return;
            if (document.getElementById('__agent_cursor__')) return;
            const el = document.createElement('div');
            el.id = '__agent_cursor__';
            el.style.cssText = `
                position: fixed;
                pointer-events: none;
                z-index: 2147483647;
                display: none;
                width: 32px;
                height: 32px;
                transition: left 0.12s ease, top 0.12s ease;
            `;
            el.innerHTML = `
                <svg width="32" height="32" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="ag" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%"   stop-color="#e0aaff"/>
                            <stop offset="100%" stop-color="#7b2ff7"/>
                        </linearGradient>
                        <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
                            <feGaussianBlur in="SourceGraphic" stdDeviation="2.5" result="blur"/>
                            <feMerge>
                                <feMergeNode in="blur"/>
                                <feMergeNode in="blur"/>
                                <feMergeNode in="SourceGraphic"/>
                            </feMerge>
                        </filter>
                    </defs>
                    <path d="M 3 2.5 L 3 20.5 L 19 11.5 Z"
                          fill="url(#ag)"
                          stroke="white" stroke-width="3.5"
                          stroke-linejoin="round" stroke-linecap="round"
                          paint-order="stroke fill"
                          filter="url(#glow)"/>
                </svg>`;
            document.body.appendChild(el);
        }""")

    async def _move_cursor(self, x: float, y: float):
        """Move the injected cursor to (x, y) — hotspot is at the arrow tip (top-left)."""
        await self._page.evaluate(f"""() => {{
            const el = document.getElementById('__agent_cursor__');
            if (!el) return;
            el.style.display = 'block';
            el.style.left = '{x}px';
            el.style.top = '{y}px';
        }}""")

    async def hover(self, index: int, pause_ms: int = 600):
        """Move the visible cursor to an element without clicking — for demos."""
        if index not in self._selector_map:
            return
        try:
            x, y = await self._get_element_center(index)
            await self._page.wait_for_timeout(400)  # let scroll settle
            await self._ensure_cursor()
            await self._move_cursor(x, y)
            print(f"[cursor] hover index {index} at ({x:.0f}, {y:.0f})")
            await self._page.wait_for_timeout(pause_ms)
        except Exception as e:
            print(f"[cursor] hover failed for index {index}: {e}")

    async def _cdp_click(self, x: float, y: float):
        print(f"[mouse] click at ({x:.0f}, {y:.0f})")
        for event_type in ("mousePressed", "mouseReleased"):
            await self._cdp.send("Input.dispatchMouseEvent", {
                "type": event_type,
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": 1,
            })

    async def _get_href(self, backend_node_id: int) -> str | None:
        try:
            node_obj = await self._cdp.send("DOM.resolveNode", {"backendNodeId": backend_node_id})
            obj_id = node_obj["object"]["objectId"]
            result = await self._cdp.send("Runtime.callFunctionOn", {
                "objectId": obj_id,
                "functionDeclaration": "function() { return this.href || null; }",
                "returnByValue": True,
            })
            return result["result"].get("value")
        except Exception:
            return None

    async def _click(self, args: dict) -> tuple[str, list]:
        index = args.get("index")

        if index not in self._selector_map:
            state = await self._page_state()
            return f"[Click failed: index {index} not in snapshot — take a fresh snapshot first]\n{state}", []

        backend_node_id = int(self._selector_map[index])
        url_before = self._page.url

        # Scroll into view, get viewport coords, show cursor, then click
        try:
            x, y = await self._get_element_center(index)
            await self._page.wait_for_timeout(400)  # let smooth scroll settle

            if self.visible_mouse:
                await self._ensure_cursor()
                await self._move_cursor(x, y)
                print(f"[cursor] → index {index} at ({x:.0f}, {y:.0f})")
                await self._page.wait_for_timeout(1400)  # pause so you can see what it's about to click

            await self._cdp_click(x, y)
            await self._page.wait_for_timeout(600)
        except Exception as e:
            print(f"[cursor] coordinate click failed: {e}")

        # If URL didn't change, the coordinate click did nothing — try href navigation
        if self._page.url == url_before:
            href = await self._get_href(backend_node_id)
            if href and href != url_before:
                print(f"[cursor] navigating via href: {href}")
                await self._page.goto(href, wait_until="domcontentloaded")
                await self._page.wait_for_timeout(1500)

        await self._attach_cdp()
        if self.visible_mouse:
            await self._ensure_cursor()  # re-inject after any navigation
        state = await self._page_state()
        return state, []

    async def _type(self, args: dict) -> tuple[str, list]:
        index = args.get("index")
        text = args.get("text", "")

        if index not in self._selector_map:
            state = await self._page_state()
            return f"[Type failed: index {index} not in snapshot — take a fresh snapshot first]\n{state}", []

        try:
            x, y = await self._get_element_center(index)
            print(f"[mouse] type target index {index} at ({x:.0f}, {y:.0f}) — text: {text!r}")
            if self.visible_mouse:
                await self._ensure_cursor()
                await self._move_cursor(x, y)
                await self._page.wait_for_timeout(400)
            await self._cdp_click(x, y)
            await self._page.wait_for_timeout(100)
            await self._page.keyboard.type(text)
            state = await self._page_state()
            return state, []
        except Exception as e:
            print(f"[mouse] type failed at index {index}: {e}")
            state = await self._page_state()
            return f"[Type failed: {e}]\n{state}", []

    async def _press_key(self, args: dict) -> tuple[str, list]:
        await self._page.keyboard.press(args.get("key", ""))
        await self._page.wait_for_timeout(400)
        state = await self._page_state()
        return state, []

    async def _scroll(self, args: dict) -> tuple[str, list]:
        direction = args.get("direction", "down")
        delta_y = 400 if direction == "down" else (-400 if direction == "up" else 0)
        delta_x = 400 if direction == "right" else (-400 if direction == "left" else 0)
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
            await self._attach_cdp()
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
                await self._attach_cdp()
                state = await self._page_state()
                return f"Switched to tab {idx}.\n{state}", []
            return f"[Invalid tab index: {idx}]", []

        elif action == "close":
            if len(self._pages) > 1:
                page = self._pages.pop(self._page_idx)
                await page.close()
                self._page_idx = min(self._page_idx, len(self._pages) - 1)
                await self._attach_cdp()
                state = await self._page_state()
                return f"Closed tab.\n{state}", []
            return "[Cannot close the last remaining tab]", []

        return f"[Unknown tab action: {action}]", []

    async def _back(self, args: dict) -> tuple[str, list]:
        await self._page.go_back()
        await self._attach_cdp()
        state = await self._page_state()
        return state, []

    async def _forward(self, args: dict) -> tuple[str, list]:
        await self._page.go_forward()
        await self._attach_cdp()
        state = await self._page_state()
        return state, []
