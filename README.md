# LangGraph MCP on AWS Bedrock

Integrate LangGraph with AWS Bedrock for building agents with MCP capabilities.

## Description

This repo implements a sample conversational agent built with LangGraph that uses AWS Bedrock and integrates with MCP for tool usage. 

<img width="547" alt="image" src="https://github.com/user-attachments/assets/3dfc27b6-f372-4058-9363-446fabb9aafe" />

> Mermaid transformed on [https://excalidraw.com/](https://excalidraw.com/) website.

## Features

- Integration with AWS Bedrock LLM models
- Tool usage through Model Context Protocol (MCP)
- Structured conversation workflow using LangGraph
- Flexible configuration options

## Requirements

- Python 3.12+
- uv tool
- AWS credentials configured
- LangChain and LangGraph libraries
- Access to AWS Bedrock models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LangGraph-MCP-on-AWS-Bedrock.git
cd LangGraph-MCP-on-AWS-Bedrock

# Set up a virtual environment
uv venv myvenv --python 3.12
source myvenv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

Run the script with default options:

```bash
python langgraph_mcp_bedrock.py

# or using uv run
```

<img width="1040" alt="image" src="https://github.com/user-attachments/assets/e99509c1-3d27-404c-ad54-5f97978bfc54" />


Or customize execution:

```bash
python langgraph_mcp_bedrock.py --question "tell me what is aws sagemaker lakehouse"
```

<img width="1042" alt="image" src="https://github.com/user-attachments/assets/75264416-c1c6-4202-90de-1d121ec7d3e6" />


## Command Line Options

- `--question`: Input question (default: "Hi there!")
- `--model`: Bedrock model ID (default: "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
- `--graph`: Display the graph structure
- `--mcp-config`: Path to custom MCP config JSON file

## Configuration

The application will look for MCP configuration in the following order:
1. Custom path specified with `--mcp-config`
2. `~/mcp.json`
3. `~/.aws/amazonq/mcp.json`
4. Default sample configuration (AWS Documentation MCP server)

## Architecture

The system follows a three-node workflow:
1. **Agent Node**: Processes the input and decides whether to use tools or provide a final response
2. **Tool Node**: Executes requested tools via MCP
3. **Final Node**: Formats and returns the final answer
