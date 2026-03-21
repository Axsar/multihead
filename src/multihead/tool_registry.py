"""Tool registry for the Agentic Core."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine


@dataclass
class ToolSpec:
    """Specification for a registered tool."""
    name: str
    description: str
    params_schema: dict[str, Any] = field(default_factory=dict)
    returns_schema: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    enabled: bool = True


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool: str
    success: bool
    output: Any = None
    error: str | None = None


class ToolRegistry:
    """Manages tool registration and execution."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, ToolResult]]] = {}
        self._register_builtins()

    def register(
        self,
        spec: ToolSpec,
        handler: Callable[..., Coroutine[Any, Any, ToolResult]],
    ) -> None:
        """Register a tool with its handler."""
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def list_tools(self) -> list[ToolSpec]:
        """Return enabled tool specs (this feeds the LLM system prompt)."""
        return [s for s in self._specs.values() if s.enabled]

    def list_all_tools(self) -> list[ToolSpec]:
        """Return ALL tool specs including disabled (for /tools list display)."""
        return list(self._specs.values())

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def validate_params(self, tool_name: str, params: dict[str, Any]) -> list[str]:
        """Validate params against the tool's schema. Return list of errors (empty = valid)."""
        spec = self._specs.get(tool_name)
        if spec is None:
            return [f"Unknown tool: {tool_name}"]

        errors: list[str] = []
        schema = spec.params_schema

        # Check required params are present
        for param_name, param_def in schema.items():
            if isinstance(param_def, dict) and param_def.get("required") and param_name not in params:
                errors.append(f"Missing required parameter: {param_name}")

        # Check no unknown params
        if schema:
            for param_name in params:
                if param_name not in schema:
                    errors.append(f"Unknown parameter: {param_name}")

        # Check types
        type_map = {"string": str, "integer": int, "object": dict, "array": list, "boolean": bool}
        for param_name, value in params.items():
            param_def = schema.get(param_name)
            if isinstance(param_def, dict):
                expected_type = param_def.get("type")
                if expected_type and expected_type in type_map:
                    if not isinstance(value, type_map[expected_type]):
                        errors.append(
                            f"Parameter '{param_name}' expected type {expected_type}, "
                            f"got {type(value).__name__}"
                        )

        return errors

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given params."""
        if tool_name not in self._handlers:
            return ToolResult(tool=tool_name, success=False, error=f"Unknown tool: {tool_name}")

        spec = self._specs.get(tool_name)
        if spec and not spec.enabled:
            return ToolResult(tool=tool_name, success=False, error=f"Tool '{tool_name}' is disabled")

        # Validate params against schema
        errors = self.validate_params(tool_name, params)
        if errors:
            return ToolResult(tool=tool_name, success=False, error=f"Validation errors: {'; '.join(errors)}")

        handler = self._handlers[tool_name]
        try:
            return await handler(params)
        except Exception as e:
            return ToolResult(tool=tool_name, success=False, error=str(e))

    def _register_builtins(self) -> None:
        """Register built-in tools."""
        self.register(
            ToolSpec(
                name="files.read",
                description="Read a file's contents",
                params_schema={"path": {"type": "string", "required": True}},
            ),
            self._tool_files_read,
        )
        self.register(
            ToolSpec(
                name="files.write",
                description="Write content to a file",
                params_schema={
                    "path": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True},
                },
                requires_approval=True,
            ),
            self._tool_files_write,
        )
        self.register(
            ToolSpec(
                name="shell.run",
                description="Run a shell command",
                params_schema={
                    "command": {"type": "string", "required": True},
                    "timeout": {"type": "integer", "default": 30},
                },
                requires_approval=True,
            ),
            self._tool_shell_run,
        )
        self.register(
            ToolSpec(
                name="python.run",
                description="Execute a Python snippet",
                params_schema={"code": {"type": "string", "required": True}},
                requires_approval=True,
            ),
            self._tool_python_run,
        )
        self.register(
            ToolSpec(
                name="llm.call",
                description="Call an LLM head with a prompt",
                params_schema={
                    "head_id": {"type": "string", "required": True},
                    "prompt": {"type": "string", "required": True},
                },
            ),
            self._tool_llm_call,
        )
        self.register(
            ToolSpec(
                name="verify.jsonschema",
                description="Validate data against a JSON schema",
                params_schema={
                    "data": {"type": "object", "required": True},
                    "schema": {"type": "object", "required": True},
                },
            ),
            self._tool_verify_jsonschema,
        )

    # -------------------------------------------------------------------
    # Built-in tool handlers
    # -------------------------------------------------------------------

    @staticmethod
    def _normalize_path(raw: str) -> Path:
        """Convert Windows paths to WSL paths if needed (e.g. D:\\foo → /mnt/d/foo)."""
        raw = raw.strip().strip('"').strip("'")
        if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
            drive = raw[0].lower()
            rest = raw[2:].replace("\\", "/")
            return Path(f"/mnt/{drive}{rest}")
        return Path(raw)

    @staticmethod
    async def _tool_files_read(params: dict[str, Any]) -> ToolResult:
        path = ToolRegistry._normalize_path(params.get("path", ""))
        if not path.exists():
            return ToolResult(tool="files.read", success=False, error=f"File not found: {path}")
        try:
            content = path.read_text(encoding="utf-8")
            return ToolResult(tool="files.read", success=True, output=content)
        except Exception as e:
            return ToolResult(tool="files.read", success=False, error=str(e))

    @staticmethod
    async def _tool_files_write(params: dict[str, Any]) -> ToolResult:
        path = ToolRegistry._normalize_path(params.get("path", ""))
        content = params.get("content", "")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(tool="files.write", success=True, output=f"Written {len(content)} chars to {path}")
        except Exception as e:
            return ToolResult(tool="files.write", success=False, error=str(e))

    @staticmethod
    async def _tool_shell_run(params: dict[str, Any]) -> ToolResult:
        command = params.get("command", "")
        timeout = params.get("timeout", 30)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\nSTDERR:\n" + stderr.decode("utf-8", errors="replace")
            return ToolResult(
                tool="shell.run", success=proc.returncode == 0,
                output=output, error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
            )
        except asyncio.TimeoutError:
            return ToolResult(tool="shell.run", success=False, error=f"Timed out after {timeout}s")
        except Exception as e:
            return ToolResult(tool="shell.run", success=False, error=str(e))

    @staticmethod
    async def _tool_python_run(params: dict[str, Any]) -> ToolResult:
        code = params.get("code", "")
        try:
            # Use exec with captured output
            local_vars: dict[str, Any] = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            result = local_vars.get("result", local_vars.get("output", "executed"))
            return ToolResult(tool="python.run", success=True, output=str(result))
        except Exception as e:
            return ToolResult(tool="python.run", success=False, error=str(e))

    @staticmethod
    async def _tool_llm_call(params: dict[str, Any]) -> ToolResult:
        # This will be wired to head_manager in agentic_core
        return ToolResult(
            tool="llm.call", success=False,
            error="llm.call requires agentic core context",
        )

    @staticmethod
    async def _tool_verify_jsonschema(params: dict[str, Any]) -> ToolResult:
        data = params.get("data", {})
        schema = params.get("schema", {})
        try:
            # Simple type checking (no jsonschema dependency)
            expected_type = schema.get("type")
            if expected_type == "object" and not isinstance(data, dict):
                return ToolResult(tool="verify.jsonschema", success=False, error="Expected object")
            if expected_type == "array" and not isinstance(data, list):
                return ToolResult(tool="verify.jsonschema", success=False, error="Expected array")
            if expected_type == "string" and not isinstance(data, str):
                return ToolResult(tool="verify.jsonschema", success=False, error="Expected string")
            return ToolResult(tool="verify.jsonschema", success=True, output="Valid")
        except Exception as e:
            return ToolResult(tool="verify.jsonschema", success=False, error=str(e))
