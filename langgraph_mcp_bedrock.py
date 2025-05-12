import os
import argparse
import asyncio
import json
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient

# Configuration
class Config:
    DEFAULT_QUESTION = "Hi there!"
    DEFAULT_MODEL = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    MCP_SERVERS = {
        "awslabs.aws-documentation-mcp-server": {
            "command": "uv",
            "args": ["run", "awslabs.aws-documentation-mcp-server"]
        }
    }

# Data Models
class AgentState(TypedDict):
    messages: List[BaseMessage]
    agent_outcome: str
    tool_calls: List[Dict[str, Any]]
    mcp_client: Optional[MultiServerMCPClient]

# Tool Handler
class MCPToolHandler:
    def __init__(self, client: MultiServerMCPClient):
        self.client = client

    async def get_tools(self) -> List[Dict]:
        return self.client.get_tools()

    async def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        tool = next((t for t in self.client.get_tools() if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        try:
            return await tool.ainvoke(params)
        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {str(e)}")

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Run LangGraph with Bedrock and MCP')
    parser.add_argument('--question', default=Config.DEFAULT_QUESTION, help='Input question')
    parser.add_argument('--model', default=Config.DEFAULT_MODEL, help='Bedrock model ID')
    parser.add_argument('--graph', action='store_true', help='Print graph structure')
    parser.add_argument('--mcp-config', help='Path to MCP config JSON file')
    return parser.parse_args()

# Model Initialization
def init_model(model_id: str) -> ChatBedrockConverse:
    return ChatBedrockConverse(model=model_id, verbose=True)

# Agent Decision Node
async def agent_node(state: AgentState, llm: ChatBedrockConverse, tool_handler: MCPToolHandler) -> Dict:
    messages = state["messages"]
    tools = await tool_handler.get_tools()
    response = llm.invoke(messages, tools=tools, tool_choice="auto")

    tool_calls = []
    if isinstance(response.content, list):
        response_content = response.content
        tool_calls = [
            {"id": block["id"], "name": block["name"], "input": block.get("input", {})}
            for block in response_content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
    else:
        response_content = [{"text": response.content}] if response.content.strip() else []

    messages.append(AIMessage(content=response_content))
    outcome = "tool" if tool_calls else "final"
    
    return {
        "agent_outcome": outcome,
        "tool_calls": tool_calls,
        "messages": messages,
        "mcp_client": state.get("mcp_client")
    }

# Tool Node
async def tool_node(state: AgentState, tool_handler: MCPToolHandler) -> Dict:
    tool_results = []
    for call in state.get("tool_calls", []):
        try:
            result = await tool_handler.execute_tool(call["name"], call["input"])
            tool_results.append({
                "toolResult": {
                    "toolUseId": call.get("id"),
                    "status": "success",
                    "content": [{"text": json.dumps(result)}]
                }
            })
        except Exception as e:
            tool_results.append({
                "toolResult": {
                    "toolUseId": call.get("id", f"error_{hash(str(e))}"),
                    "status": "error",
                    "content": [{"text": f"Tool execution failed: {str(e)}"}]
                }
            })

    if tool_results:
        state["messages"].append(HumanMessage(content=tool_results))

    return {
        "messages": state["messages"],
        "tool_calls": [],
        "agent_outcome": "agent",
        "mcp_client": state.get("mcp_client")
    }

# Final Node
def final_node(state: AgentState) -> Dict:
    return {
        "messages": state["messages"],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "steps": len(state["messages"])
        }
    }

# Graph Construction
async def build_graph(llm: ChatBedrockConverse, tool_handler: MCPToolHandler):
    workflow = StateGraph(AgentState)

    # Create a wrapper for agent_node to bind llm and tool_handler
    async def agent_node_wrapper(state: AgentState) -> Dict:
        return await agent_node(state, llm, tool_handler)

    # Create a wrapper for tool_node to bind tool_handler
    async def tool_node_wrapper(state: AgentState) -> Dict:
        return await tool_node(state, tool_handler)

    workflow.add_node("agent", agent_node_wrapper)
    workflow.add_node("tool", tool_node_wrapper)
    workflow.add_node("final", final_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", lambda st: st["agent_outcome"], {"tool": "tool", "final": "final"})
    workflow.add_edge("tool", "agent")
    workflow.set_finish_point("final")

    return workflow.compile()

# Display Final Response
def display_final_response(messages: List[BaseMessage]) -> None:
    for msg in reversed(messages):
        if (isinstance(msg, AIMessage) and msg.content and
                isinstance(msg.content, list) and not any(
                    isinstance(block, dict) and block.get("type") == "tool_use"
                    for block in msg.content)):
            for block in msg.content:
                if isinstance(block, dict) and "text" in block:
                    print("\nFinal Response:")
                    print(block["text"])
                    return
    print("No valid final response found.")

# Main Execution Loop
async def run_loop(question: str, model_id: str, show_graph: bool = False, mcp_config: Optional[Dict] = None):
    llm = init_model(model_id)
    async with MultiServerMCPClient(mcp_config or Config.MCP_SERVERS) as client:
        tool_handler = MCPToolHandler(client)
        graph = await build_graph(llm, tool_handler)

        if show_graph:
            print("\nGraph Structure:")
            print(graph.get_graph().draw_mermaid())

        state = {"messages": [HumanMessage(content=question)], "mcp_client": client}
        final_state = None

        print("\nExecution Trace:")
        async for step in graph.astream(state):
            node, result = next(iter(step.items()))
            outcome = result.get("agent_outcome", "N/A")

            if node == "agent" and outcome == "tool":
                tool_calls = result.get("tool_calls", [])
                if tool_calls:
                    print(f"Tool calls: {tool_calls}")

            print(f"Step: {node} ---> {outcome}")
            final_state = result

        if final_state and "messages" in final_state:
            display_final_response(final_state["messages"])

# Main Entry Point
async def main():
    args = parse_args()
    
    mcp_config = None
    if args.mcp_config:
        with open(args.mcp_config) as f:
            mcp_config = json.load(f)["mcpServers"]
    else:
        for path in [os.path.expanduser(p) for p in ["~/mcp.json", "~/.aws/amazonq/mcp.json"]]:
            if os.path.exists(path):
                with open(path) as f:
                    mcp_config = json.load(f)["mcpServers"]
                break

    await run_loop(
        question=args.question,
        model_id=args.model,
        show_graph=args.graph,
        mcp_config=mcp_config or Config.MCP_SERVERS
    )

if __name__ == "__main__":
    asyncio.run(main())