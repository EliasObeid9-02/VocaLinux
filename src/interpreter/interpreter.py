import subprocess

from interpreter.error import Error, empty_err, generate_invalid_section_err
from interpreter.parse_argument import parse_argument
from interpreter.parse_capital_options import parse_captial_options
from interpreter.parse_command import parse_command
from interpreter.parse_long_options import parse_long_options
from interpreter.parse_options import parse_options


class Command:
    command: str
    short_options: str
    captial_options: str
    long_options: str
    arguments: list[str]
    is_valid: bool | None = None

    def set_command(self, command: str):
        self.command = command

    def set_short_options(self, short_options: str):
        self.short_options = "-" + short_options

    def set_capital_options(self, captial_options: str):
        self.captial_options = "-" + captial_options

    def set_long_options(self, long_options: list[str]):
        self.long_options = " ".join(["--" + option for option in long_options])

    def add_argument(self, argument: str):
        self.arguments.append(argument)

    def validate_command(self):
        is_valid = True
        # TODO: implement validation method

        self.is_valid = is_valid

    def execute_command(self):
        if self.is_valid is None or not self.is_valid:
            return

        subprocess.run(
            [
                self.command,
                self.short_options,
                self.captial_options,
                self.long_options,
                *self.arguments,
            ]
        )


def process_section(command: Command, type: str, words: list[str]) -> Error:
    match type:
        case "command":
            command_text, err = parse_command(words)
            if not err.is_empty():
                return err
            command.set_command(command_text)
        case "options":
            options_text, err = parse_options(words)
            if not err.is_empty():
                return err
            command.set_short_options(options_text)
        case "capital options":
            options_text, err = parse_captial_options(words)
            if not err.is_empty():
                return err
            command.set_capital_options(options_text)
        case "long options":
            options_text, err = parse_long_options(words)
            if not err.is_empty():
                return err
            command.set_long_options(options_text)
        case "argument":
            argument_text, err = parse_argument(words)
            if not err.is_empty():
                return err
            command.add_argument(argument_text)
        case _:
            return generate_invalid_section_err(type)
    return empty_err


def interpreter(text: str):
    start_index: int = -1
    section_type: str = ""
    command: Command = Command()

    words: list[str] = text.lower().strip().split(" ")
    for i in range(len(words)):
        if start_index != -1:
            if words[i] == "end":
                err: Error = process_section(
                    command, section_type, words[start_index:i]
                )
                section_type = ""
                start_index = -1
        else:
            match words[i]:
                case "command", "argument", "options":
                    section_type = words[i]
                    start_index = i + 1
                case _:
                    if i + 1 == len(words):
                        # TODO: handle failure
                        return

                    if words[i + 1] != "options":
                        # TODO: handle failure
                        return

                    if words[i] != "capital" and words[i] != "long":
                        # TODO: handle failure
                        return
                    section_type = f"{words[i]} {words[i + 1]}"
                    start_index = i + 2
                    i += 1

    command.validate_command()
    command.execute_command()
