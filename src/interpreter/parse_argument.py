from interpreter.error import Error, empty_err, generate_escaped_non_keyword_err

escape_keyword = "backslash"

keywords = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "dot": ".",
    "underscore": "_",
    "dash": "-",
    "tilde": "~",
    "slash": "/",
    "space": " ",
}


def is_keyword(word: str) -> bool:
    return keywords.get(word) is not None


def parse_argument(words: list[str]) -> tuple[str, Error]:
    result: str = ""
    is_escaped: bool = False
    for word in words:
        if not is_keyword(word) and is_escaped:
            return "", generate_escaped_non_keyword_err(word)

        if is_keyword(word) and not is_escaped:
            result += keywords[word]
        else:
            result += word
        is_escaped = word == escape_keyword
    return result, empty_err
