from interpreter.error import Error, empty_err, generate_invalid_short_option_err


def parse_options(words: list[str]) -> tuple[str, Error]:
    result: str = ""
    for word in words:
        # handle the case we want to use the -1 option
        if word == "one":
            result += "1"
            continue

        if len(word) > 1:
            return "", generate_invalid_short_option_err(word)
        result += word
    return result, empty_err
