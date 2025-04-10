from interpreter.error import Error, empty_err, generate_invalid_short_option_err


def parse_captial_options(words: list[str]) -> tuple[str, Error]:
    result: str = ""
    for word in words:
        if len(word) > 1:
            return "", generate_invalid_short_option_err(word)
        result += word.upper()
    return result, empty_err
