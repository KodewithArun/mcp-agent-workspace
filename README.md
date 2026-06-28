# MCP Agent Workspace

Multi-server MCP (Model Context Protocol) agent using `langchain-mcp-adapters`, `ChatOllama`, and `Streamlit`.

## Setup

```bash
uv sync
```

## Usage

### CLI Client

```bash
uv run python client1.py
```

### Streamlit UI

```bash
uv run streamlit run main.py
```

## MCP Servers

- **math** — local stdio server via `fastmcp`
- **expensetracker** — remote HTTP server at Render

Edit `SERVERS` in `main.py` / `client1.py` to add or modify servers.
