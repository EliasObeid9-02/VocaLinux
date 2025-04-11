class Error:
    error_message: str | None
    internal_error: str | None

    def is_empty(self) -> bool:
        return self.error_message is None and self.internal_error is None

    def __init__(
        self, error_message: str | None = None, internal_error: str | None = None
    ):
        self.error_message = error_message
        self.internal_error = internal_error

    def __str__(self) -> str:
        return self.error_message or "Empty error"


empty_err = Error()


def generate_invalid_command_err(text: str) -> Error:
    return Error(
        error_message="Invalid command. Please provide a supported command.",
        internal_error=f"The provided text (%s) is not in the list of supported commands"
        % text,
    )


def generate_escaped_non_keyword_err(word: str) -> Error:
    return Error(
        error_message="Tried to escape non keyword.",
        internal_error=f"The provided word (%s) is not a keyword" % word,
    )


def generate_invalid_short_option_err(word: str) -> Error:
    return Error(
        error_message="Provided option is not a letter.",
        internal_error=f"The provided word (%s) is not a letter" % word,
    )


def generate_invalid_section_err(type: str) -> Error:
    return Error(
        error_message="Provided section keyword is invalid.",
        internal_error=f"The provided keyword (%s) is not a valid section type" % type,
    )
