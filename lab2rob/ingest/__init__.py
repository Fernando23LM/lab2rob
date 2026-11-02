"""Protocol ingestion: raw published protocols -> schema Task objects."""

from .bioprot import (
    DEFAULT_TIER1_PATH,
    RawProtocol,
    load_bioprot_id,
    load_tier1_file,
)

# NOTE: protocols_io is intentionally not re-exported here: importing it at
# package init makes `python -m lab2rob.ingest.protocols_io` emit a
# RuntimeWarning (module imported under two names). Import it directly.

__all__ = ["DEFAULT_TIER1_PATH", "RawProtocol", "load_bioprot_id", "load_tier1_file"]
