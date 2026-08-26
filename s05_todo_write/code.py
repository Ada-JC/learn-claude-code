#!/usr/bin/env python3
"""
s05_todo_write.py - TodoWrite

The model tracks its progress through a TodoManager. After three rounds
without an update, the harness adds a reminder alongside the tool results.

    +----------+      +-------+      +--------------+
    |   User   | ---> |  LLM  | ---> | Tools        |
    |  prompt  |      |       |      | + todo_write |
    +----------+      +---^---+      +------+-------+
                          |                 | update
                          |          +------v----------+
                          |          | TodoManager     |
                          |          | [ ] pending     |
                          |          | [>] in progress |
                          |          | [x] completed   |
                          |          +------+----------+
                          | tool_result     |
                          +-----------------+

              rounds_since_todo >= 3 -> add <reminder>
"""

import ast
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# s05 change: SYSTEM prompt adds planning guidance
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)


# -- Tool implementations from s02-s04 --

def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        matches = sorted({
            match for match in g.glob(
                pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# -- New in s05: structured state the model updates --

class TodoManager:
    def __init__(self):
        self.items: list[dict] = []

    def update(self, todos: list | str) -> str:
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                try:
                    todos = ast.literal_eval(todos)
                except (SyntaxError, ValueError) as e:
                    raise ValueError("todos must be a list or JSON array string") from e

        if not isinstance(todos, list):
            raise ValueError("todos must be a list")
        if len(todos) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"todos[{index}] must be an object")

            content = str(todo.get("content", "")).strip()
            status = str(todo.get("status", "pending")).lower()
            if not content:
                raise ValueError(f"todos[{index}] requires content")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"todos[{index}] has invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"content": content, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one todo can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."

        lines = []
        for todo in self.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[todo["status"]]
            lines.append(f"{marker} {todo['content']}")

        done = sum(todo["status"] == "completed" for todo in self.items)
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


def run_todo_write(todos: list | str) -> str:
    try:
        output = TODO.update(todos)
    except ValueError as e:
        return f"Error: {e}"
    print(f"\n\033[33m## Current Tasks\033[0m\n{output}")
    return output

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    # s05: new tool
    {"name": "todo_write", "description": "Create and manage a task list for your current coding session.",
     "input_schema": {"type": "object", "properties": {"todos": {"type": "array", "maxItems": 20, "items": {"type": "object", "properties": {"content": {"type": "string", "minLength": 1}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}, "required": ["todos"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
}


# -- Hook system from s04 --

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
SHELL_SEPARATORS = ";&|\n"
DESTRUCTIVE_COMMANDS = {"rm", "del"}
SHELL_WRAPPERS = {"sh", "bash", "zsh", "dash", "cmd", "cmd.exe"}
COMMAND_PREFIXES = {"command", "call"}
CONTROL_PREFIXES = {"then", "do", "else", "!", "{"}
COMPARE_OPERATORS = {"equ", "neq", "lss", "leq", "gtr", "geq"}
MAX_COMMAND_NESTING = 16
DESTRUCTIVE_SUBCOMMAND = re.compile(
    r"(?i)(?:\$\(|[<>]\(|\x60)\s*(?:rm|del)"
    r"(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["> /etc/", "chmod 777"]


def shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(
        command, posix=False, punctuation_chars=SHELL_SEPARATORS
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def shell_syntax_outside_single_quotes(command: str) -> str:
    visible = []
    single_quoted = double_quoted = escaped = False
    for char in command:
        if escaped:
            visible.append(" ")
            escaped = False
        elif char == "\\" and not single_quoted:
            visible.append(" ")
            escaped = True
        elif char == '"' and not single_quoted:
            double_quoted = not double_quoted
            visible.append(char)
        elif char == "'" and not double_quoted:
            single_quoted = not single_quoted
            visible.append(" ")
        else:
            visible.append(" " if single_quoted else char)
    return "".join(visible)


def unquote_shell_token(token: str) -> str:
    if len(token) >= 2 and token[0] in "'\"" and token[-1] == token[0]:
        return token[1:-1]
    return token


def command_name(token: str) -> str:
    value = unquote_shell_token(token).lstrip("@").strip("()").casefold()
    if value.startswith("del/"):
        return "del"
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def is_shell_separator(token: str) -> bool:
    return bool(token) and all(char in SHELL_SEPARATORS for char in token)


def is_shell_assignment(token: str) -> bool:
    name, separator, _ = unquote_shell_token(token).partition("=")
    return bool(
        separator
        and name
        and not name[0].isdigit()
        and name.replace("_", "a").isalnum()
    )


def segment_has_destructive_command(
    tokens: list[str], depth: int = 0
) -> bool:
    if depth >= MAX_COMMAND_NESTING:
        return True

    index = 0
    while index < len(tokens) and is_shell_assignment(tokens[index]):
        index += 1
    if index >= len(tokens):
        return False

    name = command_name(tokens[index])
    if name in DESTRUCTIVE_COMMANDS:
        return True
    if name in CONTROL_PREFIXES:
        return segment_has_destructive_command(tokens[index + 1:], depth + 1)
    if name == "env":
        index += 1
        while index < len(tokens) and (
            unquote_shell_token(tokens[index]).startswith("-")
            or is_shell_assignment(tokens[index])
        ):
            index += 1
        return segment_has_destructive_command(tokens[index:], depth + 1)
    if name in COMMAND_PREFIXES:
        index += 1
        options = []
        while (
            index < len(tokens)
            and unquote_shell_token(tokens[index]).startswith("-")
        ):
            options.append(unquote_shell_token(tokens[index]))
            index += 1
        if name == "command" and any(
            "v" in option.lstrip("-").casefold() for option in options
        ):
            return False
        return segment_has_destructive_command(tokens[index:], depth + 1)
    if name in SHELL_WRAPPERS:
        for flag_index in range(index + 1, len(tokens)):
            flag = unquote_shell_token(tokens[flag_index]).casefold()
            is_command_flag = (
                flag in {"/c", "/k"}
                if name.startswith("cmd")
                else flag.startswith("-")
                and not flag.startswith("--")
                and "c" in flag[1:]
            )
            if is_command_flag:
                nested = " ".join(
                    unquote_shell_token(token)
                    for token in tokens[flag_index + 1:]
                )
                return contains_destructive_command(nested, depth + 1)
        return False
    if name == "if":
        index += 1
        while (
            index < len(tokens)
            and command_name(tokens[index]) in {"/i", "not"}
        ):
            index += 1
        if index >= len(tokens):
            return False
        condition = command_name(tokens[index])
        if condition in {"exist", "defined", "errorlevel", "cmdextversion"}:
            return segment_has_destructive_command(
                tokens[index + 2:], depth + 1
            )
        if "==" in unquote_shell_token(tokens[index]):
            return segment_has_destructive_command(
                tokens[index + 1:], depth + 1
            )
        if (
            index + 2 < len(tokens)
            and command_name(tokens[index + 1]) in COMPARE_OPERATORS
        ):
            return segment_has_destructive_command(
                tokens[index + 3:], depth + 1
            )
        return False
    if name == "for":
        for do_index, token in enumerate(tokens[index + 1:], index + 1):
            if command_name(token) == "do":
                return segment_has_destructive_command(
                    tokens[do_index + 1:], depth + 1
                )
    return False


def contains_destructive_command(command: str, depth: int = 0) -> bool:
    if depth >= MAX_COMMAND_NESTING:
        return True

    try:
        tokens = shell_tokens(command)
    except ValueError:
        return True
    if DESTRUCTIVE_SUBCOMMAND.search(
        shell_syntax_outside_single_quotes(command)
    ):
        return True

    segment = []
    for token in tokens:
        if is_shell_separator(token):
            if segment_has_destructive_command(segment, depth):
                return True
            segment = []
        else:
            segment.append(token)
    return segment_has_destructive_command(segment, depth)


def permission_hook(block):
    """PreToolUse: s03 permission logic, registered as an s04 hook."""
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if contains_destructive_command(command) or any(
            keyword in command for keyword in DESTRUCTIVE
        ):
            print(f"\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

def log_hook(block):
    """PreToolUse: log every tool call."""
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
    return None

def large_output_hook(block, output):
    """PostToolUse: warn on large output."""
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] Large output from {block.name}: {len(str(output))} chars\033[0m")
    return None

def context_inject_hook(query: str):
    """UserPromptSubmit: log working directory."""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None

def summary_hook(messages: list):
    """Stop: print tool call count."""
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


# -- Agent loop with the reminder counter --

def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        results = []
        used_todo = False
        for block in tool_calls:
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            handler = TOOL_HANDLERS.get(block.name)
            try:
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
            except Exception as e:
                output = f"Error: {e}"

            trigger_hooks("PostToolUse", block, output)

            if block.name == "todo_write":
                used_todo = True

            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": str(output)})

        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.append({"type": "text",
                            "text": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s05: TodoWrite - plan before execution")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s05 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
