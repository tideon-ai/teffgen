"""
Bash/Shell command execution tool with security controls.

This module provides a tool for executing shell commands with
configurable security: allowed/blocked command lists, timeout,
environment variable filtering, and working directory control.
"""

import asyncio
import logging
import os
import re
import shlex
import subprocess
from typing import Any, Dict, List, Optional, Set

from ..base_tool import (
    BaseTool,
    ToolCategory,
    ToolMetadata,
    ParameterSpec,
    ParameterType,
)

logger = logging.getLogger(__name__)

# Default blocked commands/patterns that are dangerous
DEFAULT_BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf ~/*",
    "dd",
    "mkfs",
    "fdisk",
    "parted",
    "mount",
    "umount",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
    "systemctl",
    "service",
    "iptables",
    "nft",
    "useradd",
    "userdel",
    "usermod",
    "passwd",
    "chown -R /",
    "chmod -R 777 /",
    ":(){ :|:& };:",  # fork bomb
}

# Patterns that indicate dangerous intent
DANGEROUS_PATTERNS = [
    r"rm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\s+/\s*$",  # rm -rf /
    r"rm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\s+/\*",  # rm -rf /*
    r">\s*/dev/sd",  # overwrite disk devices
    r"mkfs\.",  # format filesystems
    r"dd\s+.*of=/dev/",  # dd to devices
    r":\(\)\s*\{",  # fork bomb
    r"\|\s*sh\b",  # piping to shell (potential injection)
    r"\|\s*bash\b",  # piping to bash
    r"curl\s+.*\|\s*(sh|bash)",  # curl | sh pattern
    r"wget\s+.*\|\s*(sh|bash)",  # wget | sh pattern
    r"eval\s+",  # eval
    r"`.*`",  # command substitution in backticks
    r"\$\(.*\)",  # command substitution
]

# Sensitive environment variables to strip
SENSITIVE_ENV_VARS = {
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "SERPAPI_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "DATABASE_URL",
    "DB_PASSWORD",
    "SECRET_KEY",
    "PRIVATE_KEY",
    "SSH_PRIVATE_KEY",
}


class BashTool(BaseTool):
    """
    Shell command execution tool with security controls.

    Features:
    - Configurable allowed/blocked command lists
    - Command timeout (default: 30s)
    - Output capture (stdout + stderr)
    - Working directory control
    - Environment variable filtering (strips sensitive vars)
    - Dangerous command detection and blocking

    Security:
    - Blocks dangerous commands by default (rm -rf /, dd, mkfs, etc.)
    - Optional allowed_commands whitelist mode
    - Strips sensitive environment variables
    - Command timeout to prevent hanging
    """

    def __init__(
        self,
        allowed_commands: Optional[List[str]] = None,
        blocked_commands: Optional[List[str]] = None,
        timeout: int = 30,
        working_directory: Optional[str] = None,
        strip_env_vars: Optional[Set[str]] = None,
        allow_command_substitution: bool = False,
    ):
        """
        Initialize the Bash tool.

        Args:
            allowed_commands: If set, ONLY these commands can be run (whitelist mode).
            blocked_commands: Additional commands to block (added to defaults).
            timeout: Command timeout in seconds (default: 30).
            working_directory: Working directory for commands (default: current dir).
            strip_env_vars: Extra env vars to strip (added to defaults).
            allow_command_substitution: Allow $() and `` in commands (default: False).
        """
        super().__init__(
            metadata=ToolMetadata(
                name="bash",
                description=(
                    "Execute shell commands safely. Use this to run system commands, "
                    "list files, check system info, process text with command-line tools, "
                    "and perform other shell operations."
                ),
                category=ToolCategory.SYSTEM,
                parameters=[
                    ParameterSpec(
                        name="command",
                        type=ParameterType.STRING,
                        description="The shell command to execute",
                        required=True,
                        min_length=1,
                        max_length=2000,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "return_code": {"type": "integer"},
                        "command": {"type": "string"},
                    },
                },
                timeout_seconds=timeout,
                tags=["bash", "shell", "system", "command"],
                examples=[
                    {
                        "command": "ls -la",
                        "output": {"stdout": "total 32\ndrwxr-xr-x ...", "return_code": 0},
                    },
                    {
                        "command": "echo 'Hello World'",
                        "output": {"stdout": "Hello World\n", "return_code": 0},
                    },
                    {
                        "command": "wc -l *.py",
                        "output": {"stdout": "  42 main.py\n  18 utils.py\n  60 total", "return_code": 0},
                    },
                ],
            )
        )

        self.allowed_commands = set(allowed_commands) if allowed_commands else None
        self.blocked_commands = set(DEFAULT_BLOCKED_COMMANDS)
        if blocked_commands:
            self.blocked_commands.update(blocked_commands)
        self.timeout = timeout
        self.working_directory = working_directory
        self.strip_env_vars = SENSITIVE_ENV_VARS.copy()
        if strip_env_vars:
            self.strip_env_vars.update(strip_env_vars)
        self.allow_command_substitution = allow_command_substitution

    def _is_command_safe(self, command: str) -> tuple:
        """
        Check if a command is safe to execute.

        Returns:
            (is_safe, reason) tuple
        """
        cmd_stripped = command.strip()

        # Check blocked commands (exact match)
        for blocked in self.blocked_commands:
            if cmd_stripped == blocked or cmd_stripped.startswith(blocked + " "):
                return False, f"Command blocked: '{blocked}' is in the blocked commands list"

        # Check dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, cmd_stripped):
                # Allow command substitution if explicitly enabled
                if self.allow_command_substitution and pattern in (r"`.*`", r"\$\(.*\)"):
                    continue
                return False, f"Command blocked: matches dangerous pattern"

        # Whitelist mode: only allowed commands
        if self.allowed_commands is not None:
            try:
                parts = shlex.split(cmd_stripped)
                base_cmd = parts[0] if parts else ""
            except ValueError:
                base_cmd = cmd_stripped.split()[0] if cmd_stripped.split() else ""

            if base_cmd not in self.allowed_commands:
                return False, (
                    f"Command '{base_cmd}' not in allowed commands list. "
                    f"Allowed: {sorted(self.allowed_commands)}"
                )

        return True, ""

    def _get_safe_env(self) -> Dict[str, str]:
        """Get environment with sensitive variables stripped."""
        env = os.environ.copy()
        for var in self.strip_env_vars:
            env.pop(var, None)
        return env

    async def _execute(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: The shell command to execute.

        Returns:
            Dict with stdout, stderr, return_code, and command.
        """
        # Safety check
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            raise ValueError(f"Security: {reason}")

        # Determine working directory
        cwd = self.working_directory or os.getcwd()
        if not os.path.isdir(cwd):
            raise ValueError(f"Working directory does not exist: {cwd}")

        env = self._get_safe_env()

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            # Truncate very long outputs
            max_output = 10000
            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + "\n... (output truncated)"
            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + "\n... (output truncated)"

            result = {
                "stdout": stdout_str,
                "stderr": stderr_str,
                "return_code": proc.returncode,
                "command": command,
            }

            # Format output for agent consumption
            if proc.returncode == 0:
                return result
            else:
                # Include stderr info in the result so agent sees the error
                result["error_info"] = f"Command exited with code {proc.returncode}"
                return result

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Command timed out after {self.timeout}s: {command}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {e}")
