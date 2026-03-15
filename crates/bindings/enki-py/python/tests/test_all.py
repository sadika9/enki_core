import asyncio
from pathlib import Path

import enki_py


async def main():
    print("Testing agent creation with missing model...")
    agent = enki_py.EnkiAgent(
        name="Test Agent",
        system_prompt_preamble="You are a concise test agent.",
        model="ollama::qwen3.5:latest",
        max_iterations=2,
        workspace_home=str("./test"),
    )
    res = await agent.run("session-a", "hello crete a hello.txt with text inside Enki is best")
    print(res)


def test_create_simple_agent():
    asyncio.run(main())


if __name__ == '__main__':
    test_create_simple_agent()
