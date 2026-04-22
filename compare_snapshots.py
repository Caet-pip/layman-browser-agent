"""
compare_snapshots.py
Runs browser_snapshot (MCP) and aria_snapshot (direct Playwright) on the same URL
and prints character counts side by side.

Direct Playwright uses a persistent browser profile (.browser_profile/) so cookies
and session data survive between runs — same behaviour as the MCP server.
"""
import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright
from mcp_client import PlaywrightMCPClient

TEST_URL = "https://www.google.com/search?q=cute+shoes+for+women&udm=28"

PROFILE_DIR = Path(__file__).parent / ".browser_profile"


async def measure_mcp(url: str) -> tuple[int, float]:
    client = PlaywrightMCPClient()
    await client.connect()
    try:
        await client.call_tool("browser_navigate", {"url": url})
        t0 = time.time()
        text, _ = await client.call_tool("browser_snapshot", {})
        elapsed = time.time() - t0
        return len(text), elapsed
    finally:
        await client.close()


async def measure_direct(url: str) -> tuple[int, float, str]:
    async with async_playwright() as pw:
        # Persistent context — saves cookies/session to disk across runs
        context = await pw.chromium.launch_persistent_context(
            str(PROFILE_DIR),
            headless=False,
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)  # let JS render

        # Detect CAPTCHA — pause so user can solve it manually
        content = await page.content()
        if "detected unusual traffic" in content or "I'm not a robot" in content:
            print("\n[!] Google CAPTCHA detected — solve it in the browser window, then press Enter here...")
            input()
            await page.wait_for_timeout(2000)

        t0 = time.time()
        snapshot = await page.locator("body").aria_snapshot()
        elapsed = time.time() - t0
        await context.close()
        return len(snapshot), elapsed, snapshot


async def main():
    print(f"URL: {TEST_URL}\n")

    print("Running MCP snapshot...")
    mcp_chars, mcp_time = await measure_mcp(TEST_URL)

    print("Running direct aria_snapshot...")
    direct_chars, direct_time, snapshot = await measure_direct(TEST_URL)

    mcp_tokens = mcp_chars // 4
    direct_tokens = direct_chars // 4
    ratio = mcp_chars / direct_chars if direct_chars else float("inf")

    print(f"\n{'─'*50}")
    print(f"{'':20} {'MCP':>12} {'Direct':>12}")
    print(f"{'─'*50}")
    print(f"{'Characters':20} {mcp_chars:>12,} {direct_chars:>12,}")
    print(f"{'~Tokens (chars/4)':20} {mcp_tokens:>12,} {direct_tokens:>12,}")
    print(f"{'Snapshot time':20} {mcp_time:>11.2f}s {direct_time:>11.2f}s")
    print(f"{'─'*50}")
    print(f"MCP is {ratio:.1f}x larger than direct aria_snapshot")
    print(f"\n{'─'*50}")
    print("Direct aria_snapshot content (first 3000 chars):")
    print(f"{'─'*50}")
    print(snapshot[:3000])
    print(f"... ({direct_chars:,} chars total)")


if __name__ == "__main__":
    asyncio.run(main())
