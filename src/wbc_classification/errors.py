"""Exception types raised for user-facing problems."""


class WBCError(Exception):
    """Base class for errors the CLI reports without a traceback."""


class ConfigError(WBCError):
    """Invalid or unreadable configuration."""


class DataError(WBCError):
    """Missing, empty or unreadable dataset / image."""


class SegmentationError(WBCError):
    """No cell could be located in an image."""


class CheckpointError(WBCError):
    """Checkpoint is missing, corrupt or inconsistent."""
