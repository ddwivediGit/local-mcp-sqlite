# Build your own Local MCP Client with LangChain

This project demonstrates how to build a **local MCP (Model Context Protocol) client** using LangChain. The client connects to a local MCP server, discovers the server's tool definitions, and sends tool requests to that server for execution against SQLite.


### Setup

To sync dependencies, run:

```sh
uv sync
```

Make sure Ollama is running locally and that you have pulled the model you want to use, for example:

```sh
ollama pull llama3.2:3b
```

---

## Usage

- Start the local MCP server (for example, the included SQLite demo server):

```sh
uv run server.py --server_type=sse
```

- Run the LangChain client:

```sh
uv run ollama_client.py --transport sse
```

- The client will fetch the available MCP tools from the server at startup and print their names and descriptions.
- Then type your own requests in the terminal. The client will choose the appropriate discovered tool and call the MCP server.

## Example Prompts

You can try prompts like:

```text
Add Alice Walker, age 31, profession Data Analyst to the database.
```

```text
Fetch all records from the database.
```

```text
Show age of Alice Walker.
```

For each request, the client will:
- discover the MCP tools exposed by the server
- choose the appropriate tool
- send the tool request to the MCP server
- print the execution trace, including tool name, arguments, SQL, raw tool output, and final English response

You can also override the default server URL or Ollama model:

```sh
uv run ollama_client.py --transport sse --server-url http://127.0.0.1:8000/sse --model llama3.2:3b
```

## Project Structure

- `server.py`: local MCP server exposing SQLite-backed tools
- `ollama_client.py`: LangChain MCP client that discovers server tools and calls them
- `demo.db`: SQLite database used by the demo server

---
