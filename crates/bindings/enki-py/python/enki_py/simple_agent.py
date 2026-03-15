from __future__ import annotations

import asyncio
import inspect
import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, TypeVar

from .agent import AgentRunResult, RunContext, _PythonToolHandler, _build_parameters_json
from .enki_py import EnkiAgent as _LowLevelEnkiAgent
from .enki_py import EnkiToolSpec


DepsT = TypeVar("DepsT")


@dataclass(frozen=True)
class ExternalTool:
    name: str
    description: str
    func: Callable[..., Any]
    uses_context: bool
    parameters_json: str

    def to_uniffi_spec(self) -> EnkiToolSpec:
        return EnkiToolSpec(
            name=self.name,
            description=self.description,
            parameters_json=self.parameters_json,
        )


def plain_tool(func: Callable[..., Any], *, name: str | None = None) -> ExternalTool:
    return ExternalTool(
        name=name or func.__name__,
        description=inspect.getdoc(func) or "",
        func=func,
        uses_context=False,
        parameters_json=_build_parameters_json(func, uses_context=False),
    )


def context_tool(func: Callable[..., Any], *, name: str | None = None) -> ExternalTool:
    parameters = list(inspect.signature(func).parameters.values())
    if not parameters:
        raise TypeError(f"Tool '{func.__name__}' must accept a RunContext argument")

    return ExternalTool(
        name=name or func.__name__,
        description=inspect.getdoc(func) or "",
        func=func,
        uses_context=True,
        parameters_json=_build_parameters_json(func, uses_context=True),
    )


class SimpleAgent(Generic[DepsT]):
    def __init__(
        self,
        model: str,
        *,
        instructions: str = "",
        tools: Iterable[ExternalTool] = (),
        deps_type: type[DepsT] | None = None,
        name: str = "Simple Agent",
        max_iterations: int = 20,
        workspace_home: str | None = None,
    ) -> None:
        self.model = model
        self.instructions = instructions
        self.deps_type = deps_type
        self.name = name
        self.max_iterations = max_iterations
        self.workspace_home = workspace_home
        self._tool_map = {tool.name: tool for tool in tools}
        self._handler = _PythonToolHandler(self._tool_map)
        self._backend = _LowLevelEnkiAgent.with_tools(
            name=self.name,
            system_prompt_preamble=self.instructions,
            model=self.model,
            max_iterations=self.max_iterations,
            workspace_home=self.workspace_home,
            tools=[tool.to_uniffi_spec() for tool in self._tool_map.values()],
            handler=self._handler,
            include_builtin_tools=False,
        )

    async def run(
        self,
        user_message: str,
        *,
        deps: DepsT | None = None,
        session_id: str | None = None,
    ) -> AgentRunResult:
        session_id = session_id or f"session-{uuid.uuid4()}"
        self._handler.set_deps(deps)
        try:
            output = await self._backend.run(session_id, user_message)
        finally:
            self._handler.clear_deps()
        return AgentRunResult(output=output)

    def run_sync(
        self,
        user_message: str,
        *,
        deps: DepsT | None = None,
        session_id: str | None = None,
    ) -> AgentRunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(user_message, deps=deps, session_id=session_id))

        result_box: dict[str, AgentRunResult] = {}
        error_box: dict[str, BaseException] = {}

        def runner() -> None:
            try:
                result_box["result"] = asyncio.run(
                    self.run(user_message, deps=deps, session_id=session_id)
                )
            except BaseException as error:  # pragma: no cover
                error_box["error"] = error

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_box:
            raise error_box["error"]
        return result_box["result"]


__all__ = ["ExternalTool", "SimpleAgent", "plain_tool", "context_tool", "RunContext"]
