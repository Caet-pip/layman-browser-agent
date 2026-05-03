import asyncio
import json
from datetime import datetime

from scraper_base import WebScraperAgent    # imports the parent from the other file


class LoggingScraperAgent(WebScraperAgent):

    def __init__(self):
        super().__init__()                  # runs parent __init__, sets playwright/browser/page to None
        self.log = []                       # new attribute only this child has

    async def scrape_content(self, url):
        html = await super().scrape_content(url)    # runs full parent scrape logic
        self.log.append({                           # then quietly records it
            "url": url,
            "time": datetime.now().isoformat(),
            "size": len(html)
        })
        return html                                 # caller still just gets html back

    def save_log(self, path="scrape_log.json"):
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"Log saved to {path}")


async def main():
    agent = LoggingScraperAgent()
    await agent.init_browser()

    try:
        await agent.scrape_content("https://example.com")
        print("After first scrape:")
        print(agent.log)
        print(len(agent.log))
        print("---")

        await agent.scrape_content("https://google.com")
        print("After second scrape:")
        print(agent.log)
        print(len(agent.log))
        print("---")

        await agent.scrape_content("https://github.com")
        print("After third scrape:")
        print(agent.log)
        print(len(agent.log))
        print("---")

        print("First entry only:")
        print(agent.log[0])

        agent.save_log()

    finally:
        await agent.close()


asyncio.run(main())
