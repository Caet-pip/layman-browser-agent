"""
test_cdp_interact.py

Isolated test for CDP click and type on Google search box.
Shows exactly what happens at each step — coordinates, events, results.
"""
import asyncio
from playwright.async_api import async_playwright
from cdp_browser_client import _serialize_ax_tree

async def main():
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(".browser_profile", headless=False)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        print("Navigating to Google...")
        await page.goto("https://www.google.com", wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        cdp = await ctx.new_cdp_session(page)

        # ── Step 1: Snapshot ──────────────────────────────────────────
        print("\n── Step 1: Snapshot ──")
        result = await cdp.send("Accessibility.getFullAXTree", {})
        nodes = result.get("nodes", [])
        serialized, selector_map = _serialize_ax_tree(nodes)
        print(serialized)
        print(f"\n{len(selector_map)} interactive elements")

        # ── Step 1b: Print raw AX node for search box ─────────────────
        search_index = 8
        backend_dom_node_id = selector_map[search_index]  # now stores backendDOMNodeId
        raw_node = next((n for n in nodes if n.get("backendDOMNodeId") == backend_dom_node_id), None)
        print(f"\n── Step 1b: Raw AX node for index [{search_index}] ──")
        print(f"  backendDOMNodeId: {backend_dom_node_id}")
        print(f"  raw node: {raw_node}")

        # ── Step 2: Get box model via backendDOMNodeId ─────────────────
        print(f"\n── Step 2: DOM.getBoxModel for index [{search_index}] ──")
        try:
            box = await cdp.send("DOM.getBoxModel", {"backendNodeId": int(backend_dom_node_id)})
            content = box["model"]["content"]
            x = (content[0] + content[2]) / 2
            y = (content[1] + content[5]) / 2
            print(f"  bounding box content points: {content}")
            print(f"  center: x={x:.1f}, y={y:.1f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            await ctx.close()
            return

        # ── Step 3: Mouse click via CDP ───────────────────────────────
        print(f"\n── Step 3: Input.dispatchMouseEvent click at ({x:.1f}, {y:.1f}) ──")
        try:
            await cdp.send("Input.dispatchMouseEvent", {
                "type": "mousePressed", "x": x, "y": y,
                "button": "left", "clickCount": 1,
            })
            await cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseReleased", "x": x, "y": y,
                "button": "left", "clickCount": 1,
            })
            print("  mousePressed + mouseReleased sent")
            await page.wait_for_timeout(500)
        except Exception as e:
            print(f"  ERROR: {e}")
            await ctx.close()
            return

        # ── Step 4: Type via keyboard ─────────────────────────────────
        print("\n── Step 4: page.keyboard.type('shoes for women') ──")
        try:
            await page.keyboard.type("shoes for women")
            print("  typed successfully")
            await page.wait_for_timeout(1000)
        except Exception as e:
            print(f"  ERROR: {e}")

        # ── Step 5: Check what's in the search box ────────────────────
        print("\n── Step 5: Check search box value via JS ──")
        try:
            value = await page.evaluate("""
                () => {
                    const el = document.querySelector('input[name="q"], textarea[name="q"], input[type="search"]');
                    return el ? el.value : 'element not found';
                }
            """)
            print(f"  search box value: '{value}'")
        except Exception as e:
            print(f"  ERROR: {e}")

        # ── Step 6: Press Enter ───────────────────────────────────────
        print("\n── Step 6: Press Enter ──")
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(2000)
        print(f"  URL after enter: {page.url}")

        input("\nPress Enter to close browser...")
        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
