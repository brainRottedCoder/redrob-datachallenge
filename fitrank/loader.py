"""Candidate data loading with schema validation and streaming JSONL support."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from fitrank.models import Candidate

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = Path("data/candidate_schema.json")
DEFAULT_SAMPLE_PATH = Path("data/sample_candidates.json")

_OPTIONAL_TOP_LEVEL = ("certifications", "languages")


def _default_schema_path() -> Path:
    return DEFAULT_SCHEMA_PATH


def _load_schema(schema_path: Path) -> dict[str, Any]:
    with schema_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _warn_missing_optional_fields(raw: dict[str, Any]) -> None:
    for field_name in _OPTIONAL_TOP_LEVEL:
        if field_name not in raw:
            logger.warning(
                "Candidate %s missing optional field %r",
                raw.get("candidate_id", "<unknown>"),
                field_name,
            )


def _validate_record(raw: dict[str, Any], validator: Draft7Validator) -> None:
    errors = sorted(validator.iter_errors(raw), key=lambda err: err.path)
    if errors:
        messages = "; ".join(error.message for error in errors[:3])
        raise ValueError(messages)


def parse_candidate(
    raw: dict[str, Any],
    validator: Draft7Validator | None = None,
) -> Candidate:
    """Parse and validate a raw candidate dict into a Candidate model."""
    if validator is not None:
        _validate_record(raw, validator)
    _warn_missing_optional_fields(raw)
    return Candidate.from_dict(raw)


def load_candidates(
    path: str | Path,
    schema_path: str | Path | None = None,
) -> Iterator[Candidate]:
    """Stream candidates from JSONL or a JSON array file."""
    source = Path(path)
    schema = Path(schema_path) if schema_path else _default_schema_path()
    validator = Draft7Validator(_load_schema(schema))

    if source.suffix.lower() == ".jsonl":
        yield from _stream_jsonl(source, validator)
    elif source.suffix.lower() == ".json":
        yield from _stream_json_array(source, validator)
    else:
        raise ValueError(f"Unsupported candidate file format: {source}")


def _stream_jsonl(path: Path, validator: Draft7Validator) -> Iterator[Candidate]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                yield parse_candidate(raw, validator)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed candidate at %s line %d: %s",
                    path,
                    line_number,
                    exc,
                )


def _stream_json_array(path: Path, validator: Draft7Validator) -> Iterator[Candidate]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"Expected JSON array in {path}")

    for index, raw in enumerate(records):
        try:
            if not isinstance(raw, dict):
                raise ValueError("candidate record must be an object")
            yield parse_candidate(raw, validator)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Skipping malformed candidate at %s index %d: %s",
                path,
                index,
                exc,
            )


def load_sample(
    path: str | Path | None = None,
    schema_path: str | Path | None = None,
) -> list[Candidate]:
    """Load the development sample candidates file."""
    source = Path(path) if path else DEFAULT_SAMPLE_PATH
    return list(load_candidates(source, schema_path=schema_path))
