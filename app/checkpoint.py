"""Session checkpoint export/import: portable snapshots of collection sessions.

A checkpoint captures one session's committed state — pinned protocol version
and definition digest, received offset, raw pending tail, cumulative frame
count, the most recent chunk (offset + digest + original response) and, when
enabled, the sequence audit progress — so the session can be imported into
another SQLite database under the same ``session_id`` and keep collecting
there.

Export reads the session row together with its protocol digest in a single
SELECT, so every value comes from the same committed moment. Import verifies
the checkpoint against the target database (exact protocol version registered
with the identical definition digest) and checks the checkpoint itself for
self-consistency — tail vs. offset, last chunk triple, stored response vs.
committed state, audit block vs. the pinned protocol — before committing it
as one INSERT. Any failure rejects the whole import and leaves existing
state untouched.
"""
from __future__ import annotations

import base64
import binascii
import json
from typing import Any, NoReturn

from .audit import MIN_BIT_WIDTH, sequence_bit_width
from .errors import SessionCheckpointInvalidError
from .models import ProtocolDefinition, SessionCheckpoint, SequenceAuditState

CHECKPOINT_FORMAT_VERSION = 1

# Cumulative audit counters, compared between the checkpoint block and the
# stored last response and checked for non-negativity on import.
_AUDIT_COUNTERS = (
    "duplicates",
    "suspected_gaps",
    "backwards",
    "indeterminates",
    "missing_total",
)


def _invalid(check: str, message: str, **details: Any) -> NoReturn:
    raise SessionCheckpointInvalidError(
        message=message,
        details={"check": check, **details},
    )


def _decode_b64(check: str, field: str, value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        _invalid(check, f"{field} is not valid base64: {exc}", field=field)


def checkpoint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Assemble the export document from one consistent session row read.

    ``row`` comes from ``Database.get_session_checkpoint`` (session columns
    plus the pinned protocol's ``definition_sha256``, wide audit values
    already decoded to exact Python ints).
    """
    audit = None
    if row["audit_enabled"]:
        audit = {
            "bit_width": row["audit_bit_width"],
            "last_sequence": row["audit_last_sequence"],
            "duplicates": row["audit_duplicates"],
            "suspected_gaps": row["audit_suspected_gaps"],
            "backwards": row["audit_backwards"],
            "indeterminates": row["audit_indeterminates"],
            "missing_total": row["audit_missing_total"],
        }
    return {
        "checkpoint_version": CHECKPOINT_FORMAT_VERSION,
        "session_id": row["session_id"],
        "protocol_id": row["protocol_id"],
        "version": row["version"],
        "definition_sha256": row["definition_sha256"],
        "received_offset": row["received_offset"],
        "tail_b64": row["tail_b64"],
        "frames_total": row["frames_total"],
        "last_chunk_offset": row["last_chunk_offset"],
        "last_chunk_sha256": row["last_chunk_sha256"],
        "last_response": json.loads(row["last_response"]) if row["last_response"] else None,
        "sequence_audit": audit,
    }


def _check_audit_block(
    audit: SequenceAuditState, cp: SessionCheckpoint, proto: ProtocolDefinition
) -> None:
    """The audit block must be consistent in itself and with the protocol."""
    expected_width = sequence_bit_width(proto)
    if expected_width is None or expected_width < MIN_BIT_WIDTH:
        _invalid(
            "sequence_audit",
            "checkpoint carries sequence audit state but the pinned protocol "
            "does not support sequence auditing",
            protocol_id=cp.protocol_id,
            version=cp.version,
            min_bit_width=MIN_BIT_WIDTH,
        )
    if audit.bit_width != expected_width:
        _invalid(
            "sequence_audit",
            "audit bit width does not match the pinned protocol's sequence field",
            bit_width=audit.bit_width,
            expected_bit_width=expected_width,
        )
    if audit.last_sequence is not None and not 0 <= audit.last_sequence < (1 << audit.bit_width):
        _invalid(
            "sequence_audit",
            "last_sequence lies outside the sequence field's modulus",
            last_sequence=audit.last_sequence,
            bit_width=audit.bit_width,
        )
    for name in _AUDIT_COUNTERS:
        if getattr(audit, name) < 0:
            _invalid(
                "sequence_audit",
                f"audit counter '{name}' must not be negative",
                counter=name,
                value=getattr(audit, name),
            )
    # Every suspected gap infers at least one missing frame.
    if audit.missing_total < audit.suspected_gaps:
        _invalid(
            "sequence_audit",
            "missing_total is smaller than the suspected gap count",
            missing_total=audit.missing_total,
            suspected_gaps=audit.suspected_gaps,
        )


def validate_checkpoint(cp: SessionCheckpoint, proto: ProtocolDefinition) -> None:
    """Check one checkpoint for self-consistency before it is imported.

    ``proto`` is the target database's registered definition of the pinned
    protocol version (its digest already verified by the caller). Every
    invariant the session state satisfies in this system is re-checked here;
    any violation raises ``SessionCheckpointInvalidError`` (422) and nothing
    is written.
    """
    if cp.checkpoint_version != CHECKPOINT_FORMAT_VERSION:
        _invalid(
            "checkpoint_version",
            "unsupported checkpoint format version",
            checkpoint_version=cp.checkpoint_version,
            supported=CHECKPOINT_FORMAT_VERSION,
        )

    # The pending tail occupies [received_offset - len(tail), received_offset).
    tail_bytes = _decode_b64("tail", "tail_b64", cp.tail_b64)
    if len(tail_bytes) > cp.received_offset:
        _invalid(
            "tail",
            "pending tail extends before the start of the stream",
            tail_length=len(tail_bytes),
            received_offset=cp.received_offset,
        )

    # The last-chunk triple is committed as one state: all or nothing.
    triple = (
        cp.last_chunk_offset is not None,
        cp.last_chunk_sha256 is not None,
        cp.last_response is not None,
    )
    if any(triple) and not all(triple):
        _invalid(
            "last_chunk",
            "last_chunk_offset, last_chunk_sha256 and last_response must "
            "appear together or not at all",
            last_chunk_offset=cp.last_chunk_offset,
            last_chunk_sha256=cp.last_chunk_sha256,
            has_last_response=cp.last_response is not None,
        )

    resp = cp.last_response
    if resp is None:
        # Without a committed chunk the session must still be pristine:
        # offsets, frame count and audit progress only ever advance together
        # with a committed append.
        if cp.received_offset != 0 or cp.frames_total != 0 or tail_bytes:
            _invalid(
                "last_chunk",
                "no last chunk recorded but the session state is not pristine",
                received_offset=cp.received_offset,
                frames_total=cp.frames_total,
                tail_length=len(tail_bytes),
            )
        if cp.sequence_audit is not None and (
            cp.sequence_audit.last_sequence is not None
            or any(getattr(cp.sequence_audit, name) for name in _AUDIT_COUNTERS)
        ):
            _invalid(
                "sequence_audit",
                "no chunk committed yet but the audit state has progressed",
                last_sequence=cp.sequence_audit.last_sequence,
            )
    else:
        # A committed chunk is non-empty, so it strictly advanced the offset.
        # The triple check above guarantees last_chunk_offset is set here.
        last_chunk_offset = cp.last_chunk_offset
        if last_chunk_offset is None or not 0 <= last_chunk_offset < cp.received_offset:
            _invalid(
                "last_chunk",
                "last chunk offset must be below the received offset",
                last_chunk_offset=cp.last_chunk_offset,
                received_offset=cp.received_offset,
            )
        identity = (
            ("session_id", resp.session_id, cp.session_id),
            ("protocol_id", resp.protocol_id, cp.protocol_id),
            ("version", resp.version, cp.version),
        )
        for field, in_response, in_checkpoint in identity:
            if in_response != in_checkpoint:
                _invalid(
                    "last_response",
                    f"stored response {field} does not match the checkpoint",
                    field=field,
                    response_value=in_response,
                    checkpoint_value=in_checkpoint,
                )
        if resp.received_offset != cp.last_chunk_offset:
            _invalid(
                "last_response",
                "stored response received_offset differs from the last chunk offset",
                response_received_offset=resp.received_offset,
                last_chunk_offset=cp.last_chunk_offset,
            )
        if resp.next_offset != cp.received_offset:
            _invalid(
                "last_response",
                "stored response next_offset differs from the received offset",
                response_next_offset=resp.next_offset,
                received_offset=cp.received_offset,
            )
        if resp.frames_total != cp.frames_total:
            _invalid(
                "last_response",
                "stored response frames_total differs from the checkpoint",
                response_frames_total=resp.frames_total,
                frames_total=cp.frames_total,
            )
        if resp.frame_count != len(resp.frames) or resp.frames_total < resp.frame_count:
            _invalid(
                "last_response",
                "stored response frame counts are inconsistent",
                frame_count=resp.frame_count,
                frames=len(resp.frames),
                frames_total=resp.frames_total,
            )
        if resp.replayed:
            _invalid(
                "last_response",
                "the stored response of a committed chunk is never a replay",
            )
        # The tail inside the stored response is the committed pending tail.
        if (
            resp.tail.start != cp.received_offset - len(tail_bytes)
            or resp.tail.end != cp.received_offset
        ):
            _invalid(
                "last_response",
                "stored response tail offsets do not match the checkpoint tail",
                response_tail={"start": resp.tail.start, "end": resp.tail.end},
                tail_length=len(tail_bytes),
                received_offset=cp.received_offset,
            )
        if _decode_b64("last_response", "tail.payload_b64", resp.tail.payload_b64) != tail_bytes:
            _invalid(
                "last_response",
                "stored response tail bytes differ from the checkpoint tail",
            )
        # Frames and discards of the last response lie inside the committed stream.
        committed = [
            (item.start, item.end)
            for item in [*resp.frames, *resp.discarded]
        ]
        if any(not (0 <= start <= end <= cp.received_offset) for start, end in committed):
            _invalid(
                "last_response",
                "stored response reports frames or discards outside the "
                "committed stream",
                received_offset=cp.received_offset,
            )

    # The audit block must agree with the pinned protocol and, when a chunk
    # was committed, with the cumulative state inside the stored response.
    if cp.sequence_audit is not None:
        _check_audit_block(cp.sequence_audit, cp, proto)
        if resp is not None:
            resp_audit = resp.sequence_audit
            if resp_audit is None:
                _invalid(
                    "sequence_audit",
                    "checkpoint carries audit state but the stored response "
                    "has none",
                )
            for name in ("bit_width", "last_sequence", *_AUDIT_COUNTERS):
                if getattr(resp_audit, name) != getattr(cp.sequence_audit, name):
                    _invalid(
                        "sequence_audit",
                        f"stored response audit '{name}' differs from the "
                        "checkpoint audit state",
                        counter=name,
                        response_value=getattr(resp_audit, name),
                        checkpoint_value=getattr(cp.sequence_audit, name),
                    )
    elif resp is not None and resp.sequence_audit is not None:
        _invalid(
            "sequence_audit",
            "stored response carries audit state but the checkpoint has none",
        )
