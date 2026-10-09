import os


def secret(name, default=None):
    """An environment secret with surrounding spaces and line breaks removed.

    Keys pasted into GitHub secrets often pick up a trailing space or newline, which
    the API then rejects as a wrong key.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default
