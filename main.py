import asyncio
import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama

st.set_page_config(page_title="MCP Agent", layout="wide")

SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "/home/arun/.local/bin/uv",
        "args": [
            "run",
            "--directory",
            "/home/arun/Desktop/mathmcpserver",
            "fastmcp",
            "run",
            "main.py",
        ],
    },
    "expensetracker": {
        "transport": "http",
        "url": "https://modelcontextprotocal-arunpandeylaudari.onrender.com/mcp",
    },
}

async def run_mcp_agent(prompt, log_box):
    try:
        log_box.text("Connecting to MCP Servers...")
        client = MultiServerMCPClient(SERVERS)
        tools = await client.get_tools()

        name_tools = {tool.name: tool for tool in tools}
        
        # Display available tools
        tools_list = "\n".join([f"- {t.name}: {t.description}" for t in tools])
        log_box.text(f"Available Tools:\n{tools_list}\n\nThinking...")

        llm = ChatOllama(model="qwen3:4b", temperature=0)
        llm_with_tools = llm.bind_tools(tools)

        response = await llm_with_tools.ainvoke(prompt)
        
        if not getattr(response, "tool_calls", None):
            return response.content, None

        # Extract tool details
        tool_call = response.tool_calls[0]
        selected_tool = tool_call["name"]
        selected_tool_args = tool_call["args"]
        selected_tool_id = tool_call["id"]
        
        # Log tool details
        log_text = f"Selected tool: {selected_tool}\nArgs: {selected_tool_args}\nID: {selected_tool_id}\n\nRunning tool..."
        log_box.text(log_text)
        
        tool_result = await name_tools[selected_tool].ainvoke(selected_tool_args)
        
        log_box.text(f"{log_text}\nResult: {tool_result}\n\nGenerating final response...")
        
        tool_message_response = ToolMessage(
           content=str(tool_result),
           tool_name=selected_tool,
           tool_call_id=selected_tool_id
        )
        
        final_response = await llm_with_tools.ainvoke([prompt, response, tool_message_response])
        return final_response.content, None

    except Exception as e:
        return None, str(e)


# --- Minimal UI Setup ---
st.title("MCP Agent Terminal")

user_prompt = st.text_input("Prompt:", value="i eat momo of 200?")

if st.button("Run"):
    if user_prompt.strip():
        st.text("--- Execution Logs ---")
        log_placeholder = st.empty()
        
        # Use simple text area placeholder for logs
        status_text = log_placeholder.text("Starting...")
        
        final_answer, error = asyncio.run(run_mcp_agent(user_prompt, log_placeholder))
        
        if error:
            st.error(f"Error: {error}")
        else:
            st.text("--- Final Response ---")
            st.write(final_answer)
    else:
        st.warning("Please enter a prompt.")