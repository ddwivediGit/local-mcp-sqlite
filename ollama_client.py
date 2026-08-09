import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are an AI assistant for tool calling.

Use the MCP tools discovered from the server to help the user interact with the database.
Choose the correct discovered MCP tool and provide the exact query argument needed for that tool.
""".strip()

DEFAULT_REQUEST_TIMEOUT_SECONDS = 45


def build_connection_config(server_url: str, transport: str) -> dict:
    if transport == "stdio":
        return {
            "transport": "stdio",
            "command": str(Path(".venv/bin/python").resolve()),
            "args": [str(Path("server.py").resolve()), "--server_type=stdio"],
            "cwd": str(Path.cwd()),
            "env": {
                "PATH": os.environ.get("PATH", ""),
                "VIRTUAL_ENV": str(Path(".venv").resolve()),
            },
        }
    return {
        "transport": "sse",
        "url": server_url,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        default="sse",
        choices=["sse", "stdio"],
        help="How the client connects to the MCP server.",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/sse",
        help="SSE endpoint exposed by the local MCP server. Used only with --transport=sse.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Ollama model name to use for the agent.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Maximum time in seconds to wait for a single agent response.",
    )
    return parser.parse_args()


@asynccontextmanager
async def load_tools(server_url: str, transport: str):
    client = MultiServerMCPClient(
        {"sqlite-demo": build_connection_config(server_url, transport)}
    )
    async with client.session("sqlite-demo") as session:
        tools = await load_mcp_tools(session, server_name="sqlite-demo")
        yield tools


def build_model(model_name: str):
    return ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=2048,
        num_predict=256,
        disable_streaming="tool_calling",
    )


def format_tool_details(tools) -> str:
    lines = ["Discovered MCP tools from server:"]
    for tool in tools:
        description = (tool.description or "").strip() or "No description"
        lines.append(f"- {tool.name}: {description}")
    return "\n".join(lines)


class ToolSelection(BaseModel):
    tool_name: str = Field(description="The MCP tool name to call.")
    query: str = Field(description="The exact query string to send to the selected tool.")


def build_trace_output(trace: dict) -> str:
    lines = [
        "Execution trace",
        f"User input: {trace['user_input']}",
        f"Parsed action: {trace['action']}",
        f"Structured request: {trace['structured_request']}",
        f"MCP tool called: {trace['tool_name']}",
        f"Tool arguments: {trace['tool_args']}",
        f"SQL: {trace['sql']}",
        f"Raw tool output: {trace['raw_tool_output']}",
        "Final response:",
        trace["final_response"],
    ]
    return "\n".join(lines)


def normalize_text_blocks(tool_result) -> list[str]:
    if not isinstance(tool_result, list):
        return []
    values = []
    for item in tool_result:
        if isinstance(item, dict) and item.get("type") == "text":
            values.append(str(item.get("text", "")))
        else:
            values.append(str(item))
    return values


def format_english_result(tool_name: str, tool_args: dict, tool_result) -> str:
    if tool_name == "add_data":
        text_values = normalize_text_blocks(tool_result)
        if tool_result is True or any(value.lower() == "true" for value in text_values):
            query = tool_args.get("query", "")
            return (
                "The record was added successfully.\n"
                f"Executed query: {query}"
            )
        return "The record could not be added to the database."

    if tool_name == "read_data":
        query = tool_args.get("query", "SELECT * FROM people")
        text_values = normalize_text_blocks(tool_result)
        if not text_values:
            return (
                "The query ran successfully, but no rows were returned.\n"
                f"Executed query: {query}"
            )

        lines = [
            "Here are the rows returned from the database:",
            f"Executed query: {query}",
        ]
        if len(text_values) % 4 == 0 and len(text_values) >= 4:
            for i in range(0, len(text_values), 4):
                row = text_values[i:i + 4]
                lines.append(
                    f"- ID {row[0]}: {row[1]} is {row[2]} years old and works as {row[3]}."
                )
        else:
            for value in text_values:
                lines.append(f"- {value}")
        return "\n".join(lines)

    return (
        f"Tool `{tool_name}` completed successfully.\n"
        f"Arguments: {tool_args}\n"
        f"Result: {tool_result}"
    )


async def route_and_execute(user_input: str, tools, model_name: str, request_timeout: int):
    tool_map = {tool.name: tool for tool in tools}
    tool_details = "\n".join(
        f"- {tool.name}: {(tool.description or '').strip() or 'No description'}"
        for tool in tools
    )
    llm = build_model(model_name).with_structured_output(ToolSelection)
    messages = [
        SystemMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n"
                f"Available MCP tools:\n{tool_details}"
            )
        ),
        HumanMessage(content=user_input),
    ]
    selection = await asyncio.wait_for(
        llm.ainvoke(messages),
        timeout=request_timeout,
    )
    tool_name = selection.tool_name.strip()
    query = selection.query.strip()

    tool = tool_map.get(tool_name)
    if tool is None:
        return (
            "The model selected a tool that is not available from the MCP server.\n"
            f"Model output: {selection.model_dump()}"
        )
    if not query:
        return (
            "The model did not provide a query for the selected MCP tool.\n"
            f"Model output: {selection.model_dump()}"
        )

    tool_args = {"query": query}
    tool_result = await asyncio.wait_for(
        tool.ainvoke(tool_args),
        timeout=request_timeout,
    )
    final_response = format_english_result(tool_name, tool_args, tool_result)
    return build_trace_output(
        {
            "user_input": user_input,
            "action": tool_name,
            "structured_request": selection.model_dump(),
            "tool_name": tool_name,
            "tool_args": tool_args,
            "sql": query,
            "raw_tool_output": tool_result,
            "final_response": final_response,
        }
    )


async def chat_loop(
    server_url: str,
    transport: str,
    model_name: str,
    request_timeout: int,
) -> None:
    async with load_tools(server_url, transport) as tools:
        print(f"Connected to the MCP server over {transport}.")
        print(format_tool_details(tools))
        print("Type 'exit' or 'quit' to stop.")

        while True:
            user_input = input("Enter your message: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                return
            if not user_input:
                continue

            try:
                result = await route_and_execute(
                    user_input=user_input,
                    tools=tools,
                    model_name=model_name,
                    request_timeout=request_timeout,
                )
            except asyncio.TimeoutError:
                print(
                    "Request timed out before completion. "
                    "Try a simpler prompt or a smaller model."
                )
                continue
            except Exception as exc:
                print(f"Request failed: {exc}")
                continue

            print(result)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        chat_loop(
            args.server_url,
            args.transport,
            args.model,
            args.request_timeout,
        )
    )
