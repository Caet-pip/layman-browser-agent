import asyncio
import base64
import subprocess
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVERS = {
    "playwright": ["@playwright/mcp@latest"],
    "cdp":        ["chrome-devtools-mcp@latest", "--no-performance-crux", "--no-usage-statistics"],
}


class BrowserMCPClient:
    def __init__(self, server: str = "mcp"):
        self.server = server
        self.session: ClientSession | None = None
        self._stack = AsyncExitStack()

    async def connect(self):
        args = MCP_SERVERS.get(self.server, MCP_SERVERS["playwright"])
        server_params = StdioServerParameters(
            command="npx",
            args=args,
            stderr=subprocess.DEVNULL,  # suppress console.warn spam from Node process
        )
        transport = await self._stack.enter_async_context(stdio_client(server_params))
        self.session = await self._stack.enter_async_context(ClientSession(*transport))
        await self.session.initialize()
        print(f"[MCP] Connected to {self.server} server ({args[0]})")

    async def get_tools(self) -> list[dict]:
        """Return tools in OpenAI function-calling format."""
        result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in result.tools
        ]

    async def call_tool(self, name: str, args: dict) -> tuple[str, list[dict]]:
        """
        Execute a tool. Returns (text_result, image_parts).
        image_parts is a list of {"type": "image_url", "image_url": {"url": "data:..."}}
        """
        result = await self.session.call_tool(name, args)

        text_parts = []
        image_parts = []

        for content in result.content:
            if content.type == "text":
                text_parts.append(content.text)
            elif content.type == "image":
                mime = getattr(content, "mimeType", "image/png")
                data = content.data  # already base64
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                })

        text = "\n".join(text_parts) if text_parts else "(no text output)"
        return text, image_parts

    async def close(self):
        await self._stack.aclose()
