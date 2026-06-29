import asyncio, os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, Runner, function_tool
from agents.mcp import MCPServerSse

load_dotenv()

MCP_URL = "https://obm-mx-mcp-test-g2abexetcmbfhbec.uksouth-01.azurewebsites.net/mcp/sse"
MCP_TOKEN = os.getenv("MCP_API_TOKEN")

@function_tool
def calc(expr: str) -> str:
    import math
    allowed = "0123456789+-*/(). %"
    if any(ch not in allowed for ch in expr):
        return "Error: only basic arithmetic allowed."
    try:
        return str(eval(expr, {"__builtins__": {}}, {"math": math}, {}))
    except Exception as e:
        return f"Error: {e}"

class ReasoningOutput(BaseModel):
    plan: List[str]
    used_tools: List[str] = []
    answer_draft: str

class ReflectionOutput(BaseModel):
    issues: List[str] = []
    improved_answer: str

async def think_then_answer_then_reflect(user_query: str) -> str:
    async with MCPServerSse(
        params={"url": MCP_URL, "headers": {"Authorization": f"Bearer {MCP_TOKEN}"}},
        cache_tools_list=True,
    ) as mcp:
        reasoner = Agent(
            name="Reasoner",
            instructions=(
                "You are a reasoning agent. Create a short plan, then solve step-by-step. "
                "Use the calc tool for arithmetic. Use available MCP tools to look up "
                "product information, pricing, or inventory as needed. Put the result in answer_draft."
            ),
            tools=[calc],
            mcp_servers=[mcp],
            output_type=ReasoningOutput,
        )

        reflector = Agent(
            name="Reflector",
            instructions=(
                "You review and refine the reasoner's output. Check logic and math; if wrong, fix it. "
                "Return a cleaner improved_answer."
            ),
            tools=[calc],
            mcp_servers=[mcp],
            output_type=ReflectionOutput,
        )

        r1 = await Runner.run(reasoner, user_query)
        draft = r1.final_output_as(ReasoningOutput)
        review_prompt = (
            f"Original question:\n{user_query}\n\n"
            f"Reasoner plan: {draft.plan}\n"
            f"Reasoner draft answer:\n{draft.answer_draft}\n\n"
            "Critique and improve."
        )
        r2 = await Runner.run(reflector, review_prompt)
        final = r2.final_output_as(ReflectionOutput)
        print("\n Reasoner Plan:", draft.plan)
        print(" Draft Answer:", draft.answer_draft)
        print(" Reflection Issues:", final.issues)
        return final.improved_answer

async def main():
    print("AgentKit Think → Answer → Reflect (type 'exit' to quit)")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        result = await think_then_answer_then_reflect(q)
        print("\nAgent:", result)
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main())
