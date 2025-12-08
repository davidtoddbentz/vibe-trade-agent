import os

from dotenv import load_dotenv
from langgraph_sdk.client import get_client

load_dotenv()

agent_id = "b84e1683-d134-4b29-ae6b-571fba50bc1e"

# This must be a PAT API key tied to your user
api_key = os.getenv("LANGGRAPH_API_KEY")
api_url = "https://prod-deepagents-agent-build-d4c1479ed8ce53fbb8c3eefc91f0aa7d.us.langgraph.app"

client = get_client(
    url=api_url,
    api_key=api_key,
    headers={
        "X-Auth-Scheme": "langsmith-api-key",
    },
)


async def get_assistant(agent_id: str):
    agent = await client.assistants.get(agent_id)
    print(agent)


if __name__ == "__main__":
    import asyncio

    asyncio.run(get_assistant(agent_id))
