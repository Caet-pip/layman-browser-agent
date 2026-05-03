"""
parse_logs.py

Parse and display agent message logs in a readable format.
Reads context_window.json (what LLM saw) and state_messages.json (full history).

Usage:
    python parse_logs.py              # shows both
    python parse_logs.py --context    # context window only
    python parse_logs.py --state      # full state only
"""
import json
import argparse
from pathlib import Path

def parse_messages(msgs: list[dict], title: str):
    print(f"\n{'='*60}")
    print(f"  {title}  ({len(msgs)} messages)")
    print(f"{'='*60}")
    for i, m in enumerate(msgs):
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            names = [tc["function"]["name"] for tc in m["tool_calls"]]
            ids   = [tc["id"] for tc in m["tool_calls"]]
            print(f"  [{i}] assistant  tool_calls: {names}")
            for tc_id in ids:
                print(f"          id: {tc_id}")
        elif role == "tool":
            content_preview = str(m.get("content", ""))[:80].replace("\n", " ")
            print(f"  [{i}] tool       id: {m.get('tool_call_id')}  → {content_preview}")
        elif role == "system":
            print(f"  [{i}] system     {str(m.get('content',''))[:60]}...")
        else:
            content_preview = str(m.get("content", ""))[:80].replace("\n", " ")
            print(f"  [{i}] {role:<10} {content_preview}")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", action="store_true", help="Show context window only")
    parser.add_argument("--state",   action="store_true", help="Show full state only")
    args = parser.parse_args()

    base = Path(__file__).parent
    show_context = args.context or not args.state
    show_state   = args.state   or not args.context

    if show_context:
        path = base / "context_window.json"
        if path.exists():
            parse_messages(json.loads(path.read_text()), "CONTEXT WINDOW (what LLM saw)")
        else:
            print("context_window.json not found — run the agent first")

    if show_state:
        path = base / "state_messages.json"
        if path.exists():
            parse_messages(json.loads(path.read_text()), "STATE MESSAGES (full history)")
        else:
            print("state_messages.json not found — run the agent first")

if __name__ == "__main__":
    main()
