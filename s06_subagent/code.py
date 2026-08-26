#!/usr/bin/env python3
"""
s06_subagent.py - Subagents

The task tool runs a second agent loop with a fresh message list. Both
loops share the working directory, but only the final text returns to
the parent conversation.

    Parent agent                    Subagent
    +------------------+            +------------------+
    | messages=[...]   |            | messages=[prompt]|
    |                  |   task     |                  |
    | tool: task       | ---------> | own agent loop   |
    |                  |            | base tools only  |
    | tool_result      | <--------- | final text       |
    +------------------+            +------------------+

The subagent has no task tool, so it cannot delegate again.
"""

import os
import re
import shlex
import subprocess
from pathlib import Path

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
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

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task for focused exploration or a self-contained subtask."
)
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the given task, then return a concise final answer."
)


# -- Base tools --

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, errors="replace", timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
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
    import glob
    try:
        matches = sorted({
            match for match in glob.glob(
                pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


BASE_TOOLS = [
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
]

BASE_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# -- Hooks --

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
    """PreToolUse: block denied operations and ask about risky ones."""
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if contains_destructive_command(command) or any(
            keyword in command for keyword in DESTRUCTIVE
        ):
            print("\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"

    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print("\n\033[33m[permission] Access outside workspace\033[0m")
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
    """UserPromptSubmit: log the working directory."""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list):
    """Stop: print the number of tool results in this message list."""
    tool_count = sum(
        1
        for message in messages
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def execute_tool(block, handlers: dict) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked)

    handler = handlers.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as e:
        output = f"Error: {e}"

    trigger_hooks("PostToolUse", block, output)
    return str(output)


# -- New in s06: a nested agent loop with fresh messages --

SUB_TOOLS = list(BASE_TOOLS)
SUB_HANDLERS = dict(BASE_HANDLERS)


def extract_text(content) -> str:
    if not isinstance(content, list):
        return str(content)
    return "\n".join(
        getattr(block, "text", "")
        for block in content
        if getattr(block, "type", None) == "text"
    )


def run_subagent(prompt: str) -> str:
    print("\n\033[35m[Subagent started]\033[0m")
    messages = [{"role": "user", "content": prompt}]

    for _ in range(30):
        response = client.messages.create(
            model=MODEL,
            system=SUB_SYSTEM,
            messages=messages,
            tools=SUB_TOOLS,
            max_tokens=8000,
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
            print("\033[35m[Subagent done]\033[0m")
            return extract_text(response.content) or "(no summary)"

        results = []
        for block in tool_calls:
            output = execute_tool(block, SUB_HANDLERS)
            print(f"  \033[90m[sub] {block.name}: {output[:100]}\033[0m")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})

    print("\033[35m[Subagent stopped]\033[0m")
    return "Subagent stopped after 30 turns without a final answer."


TASK_TOOL = {
    "name": "task",
    "description": "Run a subagent with fresh conversation context and return its final text.",
    "input_schema": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "minLength": 1}},
        "required": ["prompt"],
    },
}

TOOLS = [*BASE_TOOLS, TASK_TOOL]
TOOL_HANDLERS = {**BASE_HANDLERS, "task": run_subagent}


# -- Parent agent loop --

def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
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
        for block in tool_calls:
            output = execute_tool(block, TOOL_HANDLERS)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s06: Subagent - fresh messages, final text returns")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s06 >> \001\033[0m\002")
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
