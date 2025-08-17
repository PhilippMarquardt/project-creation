from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, Literal

# ---- third-party imports
# Pydantic's BaseModel and Field for Gemini-compatible tool schemas
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver  # preferred checkpointer


# ===============================
# Workspace sandbox (path safety)
# ===============================

class WorkspaceError(RuntimeError):
    pass


@dataclass
class Workspace:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: Union[str, Path]) -> Path:
        p = (self.root / path).resolve() if not str(path).startswith(str(self.root)) else Path(path).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise WorkspaceError(f"Path escapes workspace root: {p}")
        return p

    def ensure_parent(self, p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)


# ==========================
# Common helpers
# ==========================

ALLOWED_COMMANDS = {
    "node", "npm", "pnpm", "yarn", "npx", "vite",
    "python", "pip", "uv", "pytest", "ruff", "black", "mypy",
    "esbuild", "tsc", "git",
}

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _get_ws(default_ws: Workspace, config: Optional[RunnableConfig]) -> Workspace:
    try:
        cfg = (config or {}).get("configurable", {})
        if isinstance(cfg.get("workspace"), Workspace):
            return cfg["workspace"]
        if "workspace_root" in cfg:
            return Workspace(Path(cfg["workspace_root"]))
    except Exception:
        pass
    return default_ws

def _run_command(ws: Workspace, command: str, args: List[str], workdir: str, timeout_sec: int, capture_stderr: bool, env: Dict[str, str], max_output_bytes: int) -> Dict[str, Any]:
    cmd = command.strip()
    if cmd not in ALLOWED_COMMANDS:
        return {"ok": False, "error": f"Command '{cmd}' is not allowed.", "allowed": sorted(ALLOWED_COMMANDS)}
    wd = ws._resolve(workdir)
    full = [cmd] + list(args)
    try:
        proc = subprocess.run(
            full, cwd=str(wd), env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT if capture_stderr else subprocess.PIPE,
            timeout=timeout_sec, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Timeout after {timeout_sec}s", "command": full}
    except FileNotFoundError:
        return {"ok": False, "error": f"Command not found: {cmd}"}
    out = proc.stdout or b""
    truncated = False
    if len(out) > max_output_bytes:
        out = out[:max_output_bytes]
        truncated = True
    return {
        "ok": proc.returncode == 0, "code": proc.returncode, "command": full,
        "cwd": str(wd.relative_to(ws.root)), "output": out.decode("utf-8", errors="replace"),
        "truncated": truncated,
    }

def _format_run_result(result: Dict[str, Any]) -> str:
    command_str = " ".join(result.get("command", []))
    output = [f"Command `{command_str}` executed in './{result.get('cwd', '.')}'."]
    status = "Success" if result.get("ok") else "Failure"
    output.append(f"Status: {status} (exit code {result.get('code', 'N/A')})")
    if result.get("error"):
        output.append(f"Error: {result['error']}")
    if result.get("output"):
        output.append(f"\n--- OUTPUT ---\n{result['output'].strip()}\n--------------")
        if result.get('truncated'):
            output.append("(Output was truncated)")
    return "\n".join(output)


# ==========================
# Pydantic schemas for Gemini compatibility
# ==========================

class FsSearchSchema(BaseModel):
    pattern: str = Field(description="The regular expression pattern to search for")
    root: str = Field(default=".", description="Root directory to search in") 
    glob: Optional[str] = Field(default=None, description="Glob pattern to filter files")
    ignore_dirs: Optional[List[str]] = Field(default=None, description="List of directory names to ignore")
    
class CmdRunSchema(BaseModel):
    command: str = Field(description="The command to run")
    args: Optional[List[str]] = Field(default=None, description="List of command arguments")
    workdir: str = Field(default=".", description="Working directory")
    timeout_sec: int = Field(default=300, description="Timeout in seconds")
    capture_stderr: bool = Field(default=True, description="Whether to capture stderr")
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")
    max_output_bytes: int = Field(default=1_000_000, description="Maximum output bytes")

class GitAddSchema(BaseModel):
    pathspecs: Optional[List[str]] = Field(default=None, description="List of path specifications to add")

class FsReadFunctionDefinitionsSchema(BaseModel):
    path: str = Field(description="The path to the file to read")

# ==========================
# Tool registry / builder
# ==========================

def build_mcp_tools(workspace_root: Union[str, Path]) -> List:
    default_ws = Workspace(Path(workspace_root))

    # ---- ✨ REFACTORED ✨ ----
    # All tools now use direct parameters instead of Pydantic models.
    # The `args_schema` argument has been removed from the `@tool` decorator.

    @tool
    def fs_write(path: str, content: str, create_dirs: bool = True, append: bool = False, *, config: RunnableConfig) -> str:
        """Write a UTF-8 text file inside the workspace."""
        ws = _get_ws(default_ws, config)
        p = ws._resolve(path)
        try:
            if create_dirs:
                ws.ensure_parent(p)
            mode = "a" if append else "w"
            with p.open(mode, encoding="utf-8", newline="") as f:
                bytes_written = f.write(content)
            rel_path = str(p.relative_to(ws.root))
            action = "Appended" if append else "Wrote"
            return f"Success. {action} {bytes_written} bytes to '{rel_path}'."
        except Exception as e:
            return f"Error writing to '{path}': {e}"

    @tool
    def fs_read(path: str, max_bytes: Optional[int] = None, *, config: RunnableConfig) -> str:
        """Read a UTF-8 text file from the workspace."""
        ws = _get_ws(default_ws, config)
        p = ws._resolve(path)
        try:
            data = p.read_bytes()
            truncated = False
            if max_bytes is not None and len(data) > max_bytes:
                data = data[:max_bytes]
                truncated = True
            text = data.decode("utf-8", errors="replace")
            rel_path = str(p.relative_to(ws.root))
            msg = f"Success. Read {len(data)} bytes from '{rel_path}'.\n"
            if truncated:
                msg += f"Content was truncated to {max_bytes} bytes.\n"
            msg += f"---\n{text}\n---"
            return msg
        except FileNotFoundError:
            return f"Error: File not found at '{path}'."
        except Exception as e:
            return f"Error reading from '{path}': {e}"
    
    @tool
    def fs_read_files(paths: List[str], max_bytes_per_file: Optional[int] = None, *, config: RunnableConfig) -> str:
        """Read content from multiple UTF-8 text files from the workspace."""
        ws = _get_ws(default_ws, config)
        results = []
        for path in paths:
            try:
                p = ws._resolve(path)
                data = p.read_bytes()
                truncated = False
                if max_bytes_per_file is not None and len(data) > max_bytes_per_file:
                    data = data[:max_bytes_per_file]
                    truncated = True
                text = data.decode("utf-8", errors="replace")
                rel_path = str(p.relative_to(ws.root))
                msg = f"Content of '{rel_path}' ({len(data)} bytes):\n"
                if truncated:
                    msg += f"(Content was truncated to {max_bytes_per_file} bytes)\n"
                msg += f"---\n{text}\n---"
                results.append(msg)
            except FileNotFoundError:
                results.append(f"Error: File not found at '{path}'.")
            except Exception as e:
                results.append(f"Error reading from '{path}': {e}")
        return "\n\n".join(results)

    @tool
    def fs_list(path: str = ".", glob: Optional[str] = None, recursive: bool = True, *, config: RunnableConfig) -> str:
        """List files/dirs under a path; supports glob and recursion. Automatically ignores node_modules contents."""
        ws = _get_ws(default_ws, config)
        base = ws._resolve(path)
        try:
            paths = base.rglob(glob or "*") if glob or recursive else base.iterdir()
            items = []
            for p in paths:
                rel_path = p.relative_to(ws.root)
                # Skip node_modules folder contents but show the folder itself
                path_parts = rel_path.parts
                if "node_modules" in path_parts:
                    # If it's exactly node_modules folder, show it
                    if len(path_parts) == 1 and path_parts[0] == "node_modules":
                        entry_type = "d" if p.is_dir() else "f"
                        items.append(f"{entry_type} {str(rel_path)}")
                    # Skip all contents inside node_modules
                    continue
                
                if not recursive and not glob and rel_path.parent != Path(path) and base.joinpath(rel_path.parent) != base:
                    continue
                entry_type = "d" if p.is_dir() else "f"
                items.append(f"{entry_type} {str(rel_path)}")
            if not items:
                return f"Success. Directory '{path}' is empty."
            return f"Success. Contents of './{path}':\n" + "\n".join(sorted(items))
        except Exception as e:
            return f"Error listing directory '{path}': {e}"

    @tool
    def fs_mkdir(path: str, exist_ok: bool = True, *, config: RunnableConfig) -> str:
        """Create a directory (and parents)."""
        ws = _get_ws(default_ws, config)
        p = ws._resolve(path)
        try:
            p.mkdir(parents=True, exist_ok=exist_ok)
            return f"Success. Created directory '{str(p.relative_to(ws.root))}'."
        except Exception as e:
            return f"Error creating directory '{path}': {e}"

    @tool
    def fs_delete(path: str, recursive: bool = False, *, config: RunnableConfig) -> str:
        """Delete a file or directory (optionally recursive)."""
        ws = _get_ws(default_ws, config)
        p = ws._resolve(path)
        rel_path = str(p.relative_to(ws.root))
        try:
            if not p.exists():
                return f"Error: Path '{rel_path}' does not exist."
            if p.is_dir():
                if not recursive:
                    return f"Error: Path '{rel_path}' is a directory; set recursive=true to remove."
                shutil.rmtree(p)
            else:
                p.unlink()
            return f"Success. Deleted '{rel_path}'."
        except Exception as e:
            return f"Error deleting '{rel_path}': {e}"

    @tool
    def fs_move(src: str, dst: str, overwrite: bool = False, *, config: RunnableConfig) -> str:
        """Move/rename a file or directory."""
        ws = _get_ws(default_ws, config)
        src_p = ws._resolve(src)
        dst_p = ws._resolve(dst)
        try:
            ws.ensure_parent(dst_p)
            if dst_p.exists() and not overwrite:
                return f"Error: Destination '{str(dst_p.relative_to(ws.root))}' exists; set overwrite=true."
            shutil.move(str(src_p), str(dst_p))
            return f"Success. Moved '{str(src_p.relative_to(ws.root))}' to '{str(dst_p.relative_to(ws.root))}'."
        except Exception as e:
            return f"Error moving '{src}' to '{dst}': {e}"

    @tool
    def fs_stat(path: str, *, config: RunnableConfig) -> str:
        """Get basic file metadata."""
        ws = _get_ws(default_ws, config)
        p = ws._resolve(path)
        rel_path = str(p.relative_to(ws.root))
        if not p.exists():
            return f"Error: Path '{rel_path}' does not exist."
        st = p.stat()
        kind = "directory" if p.is_dir() else "file"
        return f"Success. Stats for '{rel_path}':\nType: {kind}\nSize: {st.st_size} bytes\nModified: {time.ctime(st.st_mtime)}"

    @tool(args_schema=FsReadFunctionDefinitionsSchema)
    def fs_read_function_definitions(path: str, *, config: RunnableConfig) -> str:
        """Read function definitions and docstrings from a file."""
        ws = _get_ws(default_ws, config)
        p = ws._resolve(path)
        rel_path = str(p.relative_to(ws.root))
        if not p.exists():
            return f"Error: Path '{rel_path}' does not exist."
        try:
            content = p.read_text(encoding="utf-8")
            tree = ast.parse(content)
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    func_info = f"def {node.name}():"
                    if docstring:
                        func_info += f'\n    """{docstring}"""'
                    functions.append(func_info)
            
            if not functions:
                return f"Success. No function definitions found in '{rel_path}'."
            return f"Success. Found function definitions in '{rel_path}':\n" + "\n\n".join(functions)
        except Exception as e:
            return f"Error reading function definitions from '{rel_path}': {e}"

    @tool(args_schema=FsSearchSchema)
    def fs_search(pattern: str, root: str = ".", glob: Optional[str] = None, ignore_dirs: Optional[List[str]] = None, *, config: RunnableConfig) -> str:
        """Search files for a regex pattern with optional glob filter."""
        ws = _get_ws(default_ws, config)
        if ignore_dirs is None:
            ignore_dirs = [".git", "node_modules", ".venv", "dist", "build"]
        base = ws._resolve(root)
        try:
            regex = re.compile(pattern)
            results = []
            paths = base.rglob(glob or "*")
            for p in paths:
                rel = p.relative_to(ws.root, walk_up=True)
                if any(part in ignore_dirs for part in rel.parts) or not p.is_file():
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(text.splitlines(), 1):
                        if regex.search(line):
                            results.append(f"{str(rel)}:{i}: {line.strip()[:200]}")
                except Exception:
                    continue
            if not results:
                return f"Success. No matches found for pattern '{pattern}'."
            return f"Success. Found {len(results)} matches for '{pattern}':\n" + "\n".join(results)
        except Exception as e:
            return f"Error during search: {e}"

    @tool(args_schema=CmdRunSchema)
    def cmd_run(command: str, args: Optional[List[str]] = None, workdir: str = ".", timeout_sec: int = 300, capture_stderr: bool = True, env: Optional[Dict[str, str]] = None, max_output_bytes: int = 1_000_000, *, config: RunnableConfig) -> str:
        """Run an allow-listed command with timeout. No shell interpolation. ALLOWED_COMMANDS = {
    "node", "npm", "pnpm", "yarn", "npx", "vite",
    "python", "pip", "uv", "pytest", "ruff", "black", "mypy",
    "esbuild", "tsc", "git",
}"""
        ws = _get_ws(default_ws, config)
        # Handle mutable defaults
        if args is None: args = []
        if env is None: env = {}
        
        result = _run_command(
            ws=ws, command=command, args=args, workdir=workdir,
            timeout_sec=timeout_sec, capture_stderr=capture_stderr,
            env=env, max_output_bytes=max_output_bytes,
        )
        return _format_run_result(result)

    def _git(args: List[str], ws: Workspace, workdir: str = ".") -> Dict[str, Any]:
        return _run_command(
            ws=ws, command="git", args=args, workdir=workdir, timeout_sec=60,
            capture_stderr=True, env={}, max_output_bytes=200_000,
        )

    @tool
    def git_init(path: str = ".", *, config: RunnableConfig) -> str:
        """Initialize a git repository in the given directory."""
        ws = _get_ws(default_ws, config)
        return _format_run_result(_git(["init"], ws, workdir=path))

    @tool(args_schema=GitAddSchema)
    def git_add(pathspecs: Optional[List[str]] = None, *, config: RunnableConfig) -> str:
        """Stage files for commit (like `git add`)."""
        ws = _get_ws(default_ws, config)
        if pathspecs is None: pathspecs = ["."]
        return _format_run_result(_git(["add"] + pathspecs, ws))

    @tool
    def git_commit(message: str, author_name: Optional[str] = None, author_email: Optional[str] = None, *, config: RunnableConfig) -> str:
        """Create a git commit with an optional author."""
        ws = _get_ws(default_ws, config)
        env = {}
        if author_name:
            env["GIT_AUTHOR_NAME"] = author_name
            env["GIT_COMMITTER_NAME"] = author_name
        if author_email:
            env["GIT_AUTHOR_EMAIL"] = author_email
            env["GIT_COMMITTER_EMAIL"] = author_email
        result = _run_command(
            ws=ws, command="git", args=["commit", "-m", message], workdir=".",
            timeout_sec=60, capture_stderr=True, env=env, max_output_bytes=200_000,
        )
        return _format_run_result(result)

    @tool
    def git_status(*, config: RunnableConfig) -> str:
        """Get short git status with branch info."""
        ws = _get_ws(default_ws, config)
        return _format_run_result(_git(["status", "--porcelain", "--branch"], ws))

    return [
        fs_write, fs_read, fs_read_files, fs_list, fs_mkdir, fs_delete, fs_move,
        fs_stat, fs_read_function_definitions, fs_search, cmd_run, git_init, git_add, git_commit, git_status
    ]


# ==========================
# Agent (LangGraph prebuilt)
# ==========================

SYSTEM_PROMPT_TEMPLATE = """\
You are an ai coding assistant. You are given a task and a list of tools to use to complete the task.
"""

class AgentState(TypedDict):
    messages: List[BaseMessage]

def pre_model(input_data):
    print(input_data["messages"][-1].content)
    return input_data

def build_mcp_agent(model: BaseChatModel, tools: List, system_prompt: str = SYSTEM_PROMPT_TEMPLATE) -> Runnable:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    checkpointer = InMemorySaver()
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
        name="mcp-react-agent",
        pre_model_hook=pre_model,
    )
    return agent


def mcp(model: BaseChatModel, workspace_root: Union[str, Path]) -> Runnable:
    tools = build_mcp_tools(workspace_root)
    agent = build_mcp_agent(model, tools)
    return agent.with_config({"configurable": {"workspace_root": str(Path(workspace_root).resolve())}})
