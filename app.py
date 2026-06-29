import asyncio
import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, Runner, function_tool
from agents.mcp import MCPServerSse

load_dotenv()

# Works locally (.env) and on Streamlit Cloud (dashboard secrets)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
MCP_TOKEN = st.secrets.get("MCP_API_TOKEN", os.getenv("MCP_API_TOKEN", ""))
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

MCP_URL = "https://obm-mx-mcp-test-g2abexetcmbfhbec.uksouth-01.azurewebsites.net/mcp/sse"


@function_tool
def calc(expr: str) -> str:
    import math
    allowed = "0123456789+-*/(). %"
    if any(ch not in allowed for ch in expr):
        return "Error: only basic arithmetic allowed."
    try:
        return str(eval(expr, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"Error: {e}"


class ReasoningOutput(BaseModel):
    plan: List[str]
    used_tools: List[str] = []
    answer_draft: str


class ReflectionOutput(BaseModel):
    issues: List[str] = []
    improved_answer: str


async def run_agent(user_query: str):
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
        return draft, final


# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Think-Then-Answer Agent", page_icon="🤖")
st.title("Testing Agent that can -> Think -> Answer & Reflect")
st.caption("Powered by OpenAI Agents SDK + private MCP tools")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "reasoning" in msg:
            with st.expander("Reasoning steps"):
                st.write("**Plan:**", msg["reasoning"]["plan"])
                st.write("**Draft:**", msg["reasoning"]["draft"])
                if msg["reasoning"]["issues"]:
                    st.write("**Issues found:**", msg["reasoning"]["issues"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            draft, final = asyncio.run(run_agent(prompt))
        st.markdown(final.improved_answer)
        with st.expander("Reasoning steps"):
            st.write("**Plan:**", draft.plan)
            st.write("**Draft:**", draft.answer_draft)
            if final.issues:
                st.write("**Issues found:**", final.issues)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final.improved_answer,
        "reasoning": {
            "plan": draft.plan,
            "draft": draft.answer_draft,
            "issues": final.issues,
        },
    })
