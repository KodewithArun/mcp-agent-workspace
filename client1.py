import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama



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



async def main():
    client = MultiServerMCPClient(SERVERS)

    tools = await client.get_tools()

    name_tools = {}
    for tool in tools:
        name_tools[tool.name] = tool
        print(f"Tool name: {tool.name}, description: {tool.description}")

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
    )

    llm_with_tools = llm.bind_tools(tools)

    prompt = "i eat momo of 200?"

    # Send the prompt to the LLM
    response = await llm_with_tools.ainvoke(prompt)
    
    if not getattr(response, "tool_calls", None):
        print(f"LLM response: {response.content}")
        return 
 
    selected_tool = response.tool_calls[0]["name"]
    selected_tool_args = response.tool_calls[0]["args"]
    selected_tool_id = response.tool_calls[0]["id"]
    
    
    
    print(f"Selected tool: {selected_tool}")
    print(f"Selected tool args: {selected_tool_args}")
    print(f"Selected tool id: {selected_tool_id}")
    
    print("================================================================")
    
    tool_result = await name_tools[selected_tool].ainvoke(selected_tool_args)
    
    print(f"Tool result: {tool_result}")
    
    ToolMessageResponse = ToolMessage(
       content=tool_result,
       tool_name=selected_tool,
       tool_call_id=selected_tool_id
    )
    
    final_response = await llm_with_tools.ainvoke([prompt, response, ToolMessageResponse])
    
    print(f"Final response: {final_response.content}")


# Run the application
if __name__ == "__main__":
    asyncio.run(main())