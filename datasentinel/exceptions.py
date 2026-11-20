"""Domain exception hierarchy.

Callers (the Streamlit UI today, FastAPI and LangGraph agents later) can catch
:class:`DataSentinelError` to handle every expected failure uniformly, or a more
specific subclass to give targeted feedback.
"""


class DataSentinelError(Exception):
    """Base class for all expected DataSentinel errors."""


class IngestionError(DataSentinelError):
    """A dataset could not be ingested."""


class UnsupportedFileTypeError(IngestionError):
    """The file extension or content type is not supported."""


class FileSizeError(IngestionError):
    """The file exceeds the configured size limit."""


class EmptyDatasetError(IngestionError):
    """The file is empty or contains no usable rows / columns."""


class DatasetParseError(IngestionError):
    """The file content is malformed and could not be parsed."""


class SchemaContractError(DataSentinelError):
    """A user-supplied schema contract (expected schema) is invalid."""
