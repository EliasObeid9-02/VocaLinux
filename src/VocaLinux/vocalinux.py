"""
This module contains the main function for interpreting spoken commands.
"""

from VocaLinux.interpreter.path_resolver import PathResolver
from VocaLinux.interpreter.translator import Translator


def interpret_command(transcript: str) -> str:
    """
    Interprets a transcript of a spoken command.

    Args:
        transcript: The transcript of the spoken command.

    Returns:
        The interpreted command as a string.
    """

    translator = Translator()
    translated_words = translator.translate(transcript.split())

    command_parts = []
    current_part = ""
    in_quote = False
    for word in translated_words:
        if word == '"':
            in_quote = not in_quote
            current_part += word
        elif word == " " and not in_quote:
            if current_part:
                command_parts.append(current_part)
            current_part = ""
        else:
            current_part += word
    if current_part:
        command_parts.append(current_part)

    path_resolver = PathResolver()
    resolved_parts = path_resolver.resolve(command_parts)
    return " ".join(resolved_parts)


if __name__ == "__main__":
    # Example 1: mkdir with a path and flags
    transcript_1 = "m k d i r space hyphen p space slash home slash user slash new slash project"
    command_1 = interpret_command(transcript_1)
    print(f"Transcript: '{transcript_1}'")
    print(f"Command: '{command_1}'")
    print("-" * 20)

    # Example 2: touch with a complex filename in quotes
    transcript_2 = "touch space double quote my file with spaces dot t x t double quote"
    command_2 = interpret_command(transcript_2)
    print(f"Transcript: '{transcript_2}'")
    print(f"Command: '{command_2}'")
    print("-" * 20)

    # Example 3: echo with redirection
    transcript_3 = (
        "echo space double quote some text double quote space greater than space file dot t x t"
    )
    command_3 = interpret_command(transcript_3)
    print(f"Transcript: '{transcript_3}'")
    print(f"Command: '{command_3}'")
    print("-" * 20)

    # Example 4: rm with flags and underscore
    transcript_4 = "r m space hyphen r f space slash temp slash old underscore stuff"
    command_4 = interpret_command(transcript_4)
    print(f"Transcript: '{transcript_4}'")
    print(f"Command: '{command_4}'")
    print("-" * 20)

    # Example 5: cp with relative paths
    transcript_5 = "c p space dot slash papers space dot dot slash speech underscore to underscore linux underscore commands"
    command_5 = interpret_command(transcript_5)
    print(f"Transcript: '{transcript_5}'")
    print(f"Command: '{command_5}'")
    print("-" * 20)

    # Example 6: ls with multiple flags
    transcript_6 = "l s space hyphen l a h"
    command_6 = interpret_command(transcript_6)
    print(f"Transcript: '{transcript_6}'")
    print(f"Command: '{command_6}'")
    print("-" * 20)

    # Example 7: Escaping a multi-word keyword
    transcript_7 = "echo space backslash double quote"
    command_7 = interpret_command(transcript_7)
    print(f"Transcript: '{transcript_7}'")
    print(f"Command: '{command_7}'")
