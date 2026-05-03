"""
test_cdp_snapshot.py

Test 2: Raw CDP Accessibility.getFullAXTree() extraction
- Launches browser via Playwright (just for process management)
- Opens a CDP session directly
- Calls Accessibility.getFullAXTree() ourselves
- Builds our own clean serialization
- Shows output + token count vs aria_snapshot()
"""
import asyncio
from playwright.async_api import async_playwright

URL = "https://www.amazon.com"

INTERACTIVE_ROLES = {
    "link", "button", "textbox", "searchbox", "combobox",
    "checkbox", "radio", "menuitem", "tab", "listbox",
    "option", "spinbutton", "slider", "switch", "treeitem",
}

SKIP_ROLES = {
    "none", "presentation", "generic", "group", "region",
    "navigation", "main", "banner", "contentinfo", "complementary",
    "separator", "img", "StaticText", "",
}

def count_tokens(text: str) -> int:
    return len(text) // 4

def serialize_ax_tree(nodes: list[dict]) -> str:
    """
    Build an id->node lookup, find root, walk tree top-down,
    emit one line per interactive node.
    """
    lookup = {n["nodeId"]: n for n in nodes}

    # Build parent->children map
    children_map: dict[str, list[str]] = {n["nodeId"]: [] for n in nodes}
    for node in nodes:
        for child_id in node.get("childIds", []):
            if child_id in children_map:
                children_map[node["nodeId"]].append(child_id)

    # Find root (no node claims it as child)
    all_child_ids = {cid for ids in children_map.values() for cid in ids}
    roots = [n["nodeId"] for n in nodes if n["nodeId"] not in all_child_ids]
    root_id = roots[0] if roots else nodes[0]["nodeId"]

    lines = []
    counter = [1]

    def get_prop(node: dict, name: str) -> str:
        for prop in node.get("properties", []):
            if prop.get("name") == name:
                val = prop.get("value", {})
                return str(val.get("value", ""))
        return ""

    def get_name(node: dict) -> str:
        name_obj = node.get("name", {})
        if isinstance(name_obj, dict):
            return name_obj.get("value", "")
        return str(name_obj) if name_obj else ""

    def get_role(node: dict) -> str:
        role_obj = node.get("role", {})
        if isinstance(role_obj, dict):
            return role_obj.get("value", "")
        return str(role_obj) if role_obj else ""

    def walk(node_id: str, depth: int = 0):
        node = lookup.get(node_id)
        if not node:
            return

        role = get_role(node)
        name = get_name(node)

        if role.lower() in INTERACTIVE_ROLES:
            idx = counter[0]
            counter[0] += 1
            selector_map[idx] = node.get("backendDOMNodeId")

            # build descriptor
            parts = [f"[{idx}]", role]
            if name:
                parts.append(f'"{name}"')

            placeholder = get_prop(node, "placeholder")
            if placeholder:
                parts.append(f'placeholder="{placeholder}"')

            value = get_prop(node, "value")
            if value:
                parts.append(f'value="{value}"')

            checked = get_prop(node, "checked")
            if checked:
                parts.append(f"checked={checked}")

            lines.append("  " * depth + " ".join(parts))

        for child_id in children_map.get(node_id, []):
            walk(child_id, depth + (1 if role.lower() in INTERACTIVE_ROLES else 0))

    walk(root_id)
    return "\n".join(lines)

def print_section(title: str, content: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    chars = len(content)
    tokens = count_tokens(content)
    lines = content.count("\n") + 1
    print(f"  chars: {chars:,}  |  tokens: ~{tokens:,}  |  lines: {lines:,}")
    print(f"{'─'*60}")
    print(content[:3000])
    if len(content) > 3000:
        print(f"\n... [{chars - 3000:,} more chars]")

async def main():
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            ".browser_profile",
            headless=False,
        )
        page = context.pages[0] if context.pages else await context.new_page()

        print(f"Navigating to {URL} ...")
        await page.goto(URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        # ── aria_snapshot for comparison ──────────────────────────────
        print("Taking aria_snapshot() for baseline...")
        raw_aria = await page.locator("body").aria_snapshot()

        # ── Raw CDP ───────────────────────────────────────────────────
        print("Opening CDP session...")
        cdp = await context.new_cdp_session(page)

        print("Calling Accessibility.getFullAXTree()...")
        ax_result = await cdp.send("Accessibility.getFullAXTree", {})
        nodes = ax_result.get("nodes", [])
        print(f"  CDP returned {len(nodes):,} AX nodes")

        serialized = serialize_ax_tree(nodes)

        # ── Results ───────────────────────────────────────────────────
        print_section("aria_snapshot() RAW (baseline)", raw_aria)
        print_section("CDP getFullAXTree() — our extraction", serialized)

        # ── Summary ───────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  aria_snapshot raw:  {len(raw_aria):>8,} chars  |  ~{count_tokens(raw_aria):>6,} tokens")
        print(f"  CDP extracted:      {len(serialized):>8,} chars  |  ~{count_tokens(serialized):>6,} tokens")
        reduction = (1 - len(serialized) / len(raw_aria)) * 100 if raw_aria else 0
        print(f"  Reduction vs raw:   {reduction:.1f}%")
        print(f"{'='*60}\n")

        await cdp.detach()
        await context.close()

if __name__ == "__main__":
    asyncio.run(main())
