from interpreter.error import Error, empty_err


def parse_long_options(words: list[str]) -> tuple[list[str], Error]:
    options: list[str] = []
    current_option: str = ""
    for word in words:
        if word == "comma":
            options.append(current_option)
            current_option = ""
        elif word == "dash":
            current_option += "-"
        else:
            current_option += word
    return options, empty_err
