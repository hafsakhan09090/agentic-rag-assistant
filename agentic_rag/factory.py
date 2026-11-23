"""Build an agent of the requested kind. Both kinds share the same tools, policy and result format."""
from __future__ import annotations

from .agent import Agent, LLMPlanner, ScriptedPlanner
from .tools import ToolBox


def make_agent(kind: str, retriever, tickets_path: str, provider=None):
    toolbox = ToolBox(retriever, tickets_path=tickets_path)
    if kind == "langchain":
        from .lc_agent import LangChainAgent

        return LangChainAgent(toolbox, provider)
    if kind == "loop":
        return Agent(toolbox, LLMPlanner(provider) if provider else ScriptedPlanner())
    raise ValueError(f"unknown agent kind '{kind}' (use loop or langchain)")
