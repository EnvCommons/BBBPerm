"""Test agent for BBBPerm environment."""

import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


async def main() -> None:
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    environment = or_client.environments.get(
        name="GeneralReasoning/bbbperm",
        base_url="http://localhost:8080"
    )

    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} test tasks")
    print(f"Testing first 3 tasks with model: {MODEL_NAME}\n")

    for task in tasks[:3]:
        print("=" * 70)
        task_spec = task.task_spec if hasattr(task, "task_spec") else task
        task_type = task_spec.get("task_type", "unknown")
        print(f"Task: {task_spec.get('task_id', 'unknown')} (type: {task_type})")
        print("-" * 70)

        async with environment.session(task=task, secrets={}) as session:
            prompt = await session.get_prompt()
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                prompt_text = prompt[0].text

            input_list = [{"role": "user", "content": prompt_text}]
            finished = False
            turn_count = 0

            while not finished and turn_count < 10:
                turn_count += 1
                response = await oai_client.responses.create(
                    model=MODEL_NAME, tools=tools, input=input_list,
                )

                for item in response.output:
                    input_list.append(item.model_dump())
                    if item.type == "function_call":
                        tool_result = await session.call_tool(
                            item.name, json.loads(str(item.arguments)),
                        )
                        finished = tool_result.finished
                        result_text = tool_result.blocks[0].text if tool_result.blocks else "No output"
                        print(f"\nTool: {item.name}")
                        print(f"Result: {result_text}")

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": result_text,
                        })
                        if finished:
                            print("Task completed")
                            break

                if not any(i.type == "function_call" for i in response.output):
                    break

        print()

    print("=" * 70)
    print("Testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
