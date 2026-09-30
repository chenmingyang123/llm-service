"""api-docs MCP Server —— 把 llm-service 的检索 / 问答能力暴露成标准 MCP tools。"""
__all__ = ["mcp", "main"]

from .server import main, mcp
