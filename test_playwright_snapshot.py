"""
test_playwright_snapshot.py

Test 1: aria_snapshot() raw vs post-processed
- Launches browser, navigates to Amazon
- Shows raw snapshot output + token count
- Shows filtered interactive-only output + token count
"""
import asyncio
import re
from playwright.async_api import async_playwright

URL = "https://www.amazon.com"

INTERACTIVE_ROLES = {
    "link", "button", "input", "select", "textarea",
    "checkbox", "radio", "combobox", "menuitem", "tab",
    "searchbox", "spinbutton", "slider", "switch",
}

def count_tokens(text: str) -> int:
    return len(text) // 4

def filter_snapshot(snapshot: str) -> str:
    """Keep only lines containing interactive elements."""
    lines = snapshot.splitlines()
    kept = []
    for line in lines:
        lower = line.lower()
        if any(f" {role}" in lower or f"[{role}]" in lower or lower.lstrip("- ").startswith(role) for role in INTERACTIVE_ROLES):
            kept.append(line)
    return "\n".join(kept)

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

        print("Taking snapshot...")
        raw = await page.locator("body").aria_snapshot()

        # ── Raw ───────────────────────────────────────────────────────
        print_section("RAW aria_snapshot()", raw)

        # ── Filtered ──────────────────────────────────────────────────
        filtered = filter_snapshot(raw)
        print_section("FILTERED (interactive elements only)", filtered)

        # ── Summary ───────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  Raw:      {len(raw):>8,} chars  |  ~{count_tokens(raw):>6,} tokens")
        print(f"  Filtered: {len(filtered):>8,} chars  |  ~{count_tokens(filtered):>6,} tokens")
        reduction = (1 - len(filtered) / len(raw)) * 100
        print(f"  Reduction: {reduction:.1f}%")
        print(f"{'='*60}\n")

        await context.close()

if __name__ == "__main__":
    asyncio.run(main())
