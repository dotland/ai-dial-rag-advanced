from enum import Enum

try:
    from enum import StrEnum  # Python 3.11+
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        """Fallback for Python < 3.11."""


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    AI = "assistant"
