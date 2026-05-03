"""
test_zappos.py

Demo: cursor hovers over several Nike shoes, scrolls down, hovers more, then clicks one.
"""
import asyncio
from cdp_browser_client import CDPBrowserClient

URL = "https://www.amazon.com/s?k=nike+running+shoes"


async def main():
    client = CDPBrowserClient(visible_mouse=True)
    await client.connect()

    print(f"Navigating to {URL}...")
    await client.call_tool("browser_navigate", {"url": URL})

    print("Taking snapshot...")
    snap, _ = await client.call_tool("browser_snapshot", {})

    # Find all product title links from the snapshot
    product_indices = []
    for line in snap.splitlines():
        line = line.strip()
        if line.startswith("[") and "] link" in line and "Nike" in line:
            idx = int(line.split("]")[0][1:])
            product_indices.append(idx)
            if len(product_indices) >= 8:
                break

    print(f"Found {len(product_indices)} Nike product links: {product_indices}")

    if not product_indices:
        print("No products found — check snapshot")
        await client.close()
        return

    # Hover over first 4 products
    print("\nHovering over first products...")
    for idx in product_indices[:4]:
        line = [l for l in snap.splitlines() if l.strip().startswith(f"[{idx}]")]
        label = line[0].strip() if line else f"index {idx}"
        print(f"  → {label[:70]}")
        await client.hover(idx, pause_ms=700)

    # Scroll down
    print("\nScrolling down...")
    await client.call_tool("browser_scroll", {"direction": "down"})
    await client._page.wait_for_timeout(600)
    await client.call_tool("browser_scroll", {"direction": "down"})
    await client._page.wait_for_timeout(600)

    # Hover over next batch
    print("\nHovering over more products...")
    for idx in product_indices[4:]:
        line = [l for l in snap.splitlines() if l.strip().startswith(f"[{idx}]")]
        label = line[0].strip() if line else f"index {idx}"
        print(f"  → {label[:70]}")
        await client.hover(idx, pause_ms=700)

    # Click the last hovered product
    click_idx = product_indices[-1]
    print(f"\nClicking [{click_idx}]...")
    await client.hover(click_idx, pause_ms=400)  # move cursor to it one more time
    result, _ = await client.call_tool("browser_click", {"index": click_idx})
    print(result[:200])
    print(f"Page URL: {client._page.url}")

    input("\nPress Enter to close...")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
