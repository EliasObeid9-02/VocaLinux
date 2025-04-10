from typing import cast

from interpreter.error import Error, empty_err, generate_invalid_command_err

commands = {
    "change directory": "cd",
    "copy": "cp",
    "echo": "echo",
    "list": "ls",
    "make directory": "mkdir",
    "move": "mv",
    "remove": "rm",
    "touch": "touch",
}


def parse_command(words: list[str]) -> tuple[str, Error]:
    text: str = " ".join(words)
    if commands.get(text) is None:
        return "", generate_invalid_command_err(text)
    return cast(str, commands.get(text)), empty_err
