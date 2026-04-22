import asyncio
import argparse
import os
from agent import BrowserAgent


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "openai"], help="LLM backend to use")
    parser.add_argument("--browser", choices=["direct", "playwright", "cdp"], default="direct", help="Browser client to use (default: direct)")
    args = parser.parse_args()

    agent = BrowserAgent(backend=args.backend, browser=args.browser)
    await agent.connect()

    print("Browser agent ready. Type your task, or 'exit' to quit.\n")

    try:
        while True:
            task = input("Task: ").strip()
            if not task:
                continue
            if task.lower() in ("exit", "quit"):
                break

            if agent.is_continuation(task):
                print("[Mode] continuing existing task")
            else:
                mode = agent.detect_mode(task)
                agent.set_mode(mode)

            result = await agent.run(task)
            print(f"\n{'='*60}\nTask: {task}\n{'-'*60}\n{result}\n{'='*60}\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("[Agent] Closing browser...")
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
