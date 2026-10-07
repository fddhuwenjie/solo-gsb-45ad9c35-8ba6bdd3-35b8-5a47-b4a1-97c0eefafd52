"""Tests for session checkpoint export/import between two SQLite databases.

Switching the collection service to another SQLite database must not break
ongoing collection: ``GET /sessions/{id}/checkpoint`` exports one session's
committed state (pinned protocol version + definition digest, received
offset, raw pending tail, cumulative frame count, the most recent chunk's
offset/digest/original response and, when enabled, the sequence audit
progress) captured at a single committed moment, and
``POST /sessions/checkpoint`` imports it into a target database under the
same ``session_id``.

Semantics under test:

* the export carries every piece of committed state, captured consistently;
* the import only accepts the exact pinned protocol version when it is
  already registered in the target database with the identical definition
  digest — a missing version is 404, a different definition is 409;
* tail, offsets, the stored last response and the audit block are checked
  for self-consistency; any violation is 422 SESSION_CHECKPOINT_INVALID;
* an existing session with the same id, a protocol mismatch or an invalid
  checkpoint fails the whole import and never overwrites existing state;
* after a successful import, cross-chunk decoding and the sequence audit
  continue, resending the last chunk replays the original response without
  double counting, and audit values past SQLite's signed 64-bit range stay
  exact.
"""
from __future__ import annotations

import base64
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import b64, build_logical, standard_protocol
from tests.test_sequence_audit_wide import INT64_MAX, frame64, u64_protocol

SYNC = b"\xaa\x55"


@pytest.fixture()
def pair(tmp_path):
    """Two independent service instances on two separate SQLite databases."""
    client_a = TestClient(
        create_app(Settings(db_path=str(tmp_path / "source.db"), max_decode_bytes=1 << 20))
    )
    client_b = TestClient(
        create_app(Settings(db_path=str(tmp_path / "target.db"), max_decode_bytes=1 << 20))
    )
    with client_a, client_b:
        yield client_a, client_b


def register(client, **overrides):
    r = client.post("/protocols", json=standard_protocol(**overrides))
    assert r.status_code == 201, r.text
    return r.json()


def make_session(client, audit=None, protocol_id="sensor-line", version=1):
    payload = {"protocol_id": protocol_id, "version": version}
    if audit is not None:
        payload["sequence_audit"] = audit
    r = client.post("/sessions", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def append(client, sid, offset, chunk):
    return client.post(
        "/sessions/append",
        json={"session_id": sid, "expected_offset": offset, "chunk_b64": b64(chunk)},
    )


def export(client, sid):
    r = client.get(f"/sessions/{sid}/checkpoint")
    assert r.status_code == 200, r.text
    return r.json()


def import_cp(client, checkpoint):
    return client.post("/sessions/checkpoint", json=checkpoint)


def mid_stream_checkpoint(client, audit=True):
    """A checkpoint with one committed frame and a pending tail."""
    register(client)
    sid = make_session(client, audit=audit)["session_id"] if audit else make_session(client)["session_id"]
    chunk = SYNC + build_logical(1, 3, 0) + (SYNC + build_logical(2, 3, 0))[:4]
    response = append(client, sid, 0, chunk).json()
    return sid, chunk, response, export(client, sid)


# ------------------------------------------------------------------ export ---


def test_export_fresh_session(pair):
    src, _ = pair
    reg = register(src)
    sid = make_session(src)["session_id"]

    cp = export(src, sid)
    assert cp["checkpoint_version"] == 1
    assert cp["session_id"] == sid
    assert cp["protocol_id"] == "sensor-line"
    assert cp["version"] == 1
    assert cp["definition_sha256"] == reg["definition_sha256"]
    assert cp["received_offset"] == 0
    assert cp["tail_b64"] == ""
    assert cp["frames_total"] == 0
    assert cp["last_chunk_offset"] is None
    assert cp["last_chunk_sha256"] is None
    assert cp["last_response"] is None
    assert cp["sequence_audit"] is None


def test_export_captures_committed_state(pair):
    src, _ = pair
    register(src)
    sid = make_session(src)["session_id"]
    f1 = SYNC + build_logical(1, 3, 2345)
    f2 = SYNC + build_logical(2, 3, -150)

    chunk1 = b"\x12\x34" + f1 + f2[:4]
    d1 = append(src, sid, 0, chunk1).json()

    # Mid-stream: the first 4 bytes of frame 2 are the raw pending tail.
    cp = export(src, sid)
    assert cp["received_offset"] == len(chunk1)
    assert base64.b64decode(cp["tail_b64"]) == f2[:4]
    assert cp["frames_total"] == 1
    assert cp["last_chunk_offset"] == 0
    assert cp["last_chunk_sha256"] == hashlib.sha256(chunk1).hexdigest()
    assert cp["last_response"] == d1
    assert cp["last_response"]["replayed"] is False

    chunk2 = f2[4:]
    d2 = append(src, sid, len(chunk1), chunk2).json()
    cp = export(src, sid)
    assert cp["received_offset"] == len(chunk1) + len(chunk2)
    assert cp["tail_b64"] == ""
    assert cp["frames_total"] == 2
    assert cp["last_chunk_offset"] == len(chunk1)
    assert cp["last_chunk_sha256"] == hashlib.sha256(chunk2).hexdigest()
    assert cp["last_response"] == d2
    assert cp["sequence_audit"] is None


def test_export_includes_audit_progress(pair):
    src, _ = pair
    register(src)
    sid = make_session(src, audit=True)["session_id"]
    chunk = SYNC + build_logical(1, 3, 0) + SYNC + build_logical(3, 3, 0)
    d = append(src, sid, 0, chunk).json()
    assert d["sequence_audit"]["suspected_gaps"] == 1

    cp = export(src, sid)
    audit = cp["sequence_audit"]
    assert audit == {
        "bit_width": 8,
        "last_sequence": 3,
        "duplicates": 0,
        "suspected_gaps": 1,
        "backwards": 0,
        "indeterminates": 0,
        "missing_total": 1,
    }
    assert cp["last_response"]["sequence_audit"]["missing_total"] == 1


def test_export_unknown_session_404(pair):
    src, _ = pair
    r = src.get("/sessions/does-not-exist/checkpoint")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SESSION_NOT_FOUND"


# ------------------------------------------------------------- round trip ----


def test_imported_session_resumes_collection(pair):
    src, dst = pair
    register(src)
    register(dst)
    sid = make_session(src)["session_id"]
    f1 = SYNC + build_logical(1, 3, 2345)
    f2 = SYNC + build_logical(2, 3, -150)
    chunk1 = b"\x12\x34" + f1 + f2[:4]
    d1 = append(src, sid, 0, chunk1).json()

    r = import_cp(dst, export(src, sid))
    assert r.status_code == 201, r.text
    progress = r.json()
    assert progress["session_id"] == sid
    assert progress["received_offset"] == len(chunk1)
    assert progress["next_offset"] == len(chunk1)
    assert base64.b64decode(progress["tail"]["payload_b64"]) == f2[:4]
    assert progress["frames_total"] == 1
    assert progress["last_chunk_offset"] == 0
    assert progress["last_chunk_sha256"] == hashlib.sha256(chunk1).hexdigest()
    assert dst.get(f"/sessions/{sid}").json()["frames_total"] == 1

    # Resending the most recent chunk replays the original response and
    # does not produce or count anything again.
    replay = append(dst, sid, 0, chunk1)
    assert replay.status_code == 200, replay.text
    assert replay.json() == {**d1, "replayed": True}
    assert dst.get(f"/sessions/{sid}").json()["frames_total"] == 1

    # The imported pending tail stitches with the next chunk: frame 2 is
    # completed exactly once, at session-wide cumulative offsets.
    d2 = append(dst, sid, len(chunk1), f2[4:]).json()
    assert [(f["start"], f["end"], f["sequence"]) for f in d2["frames"]] == [
        (len(chunk1) - 4, len(chunk1) - 4 + len(f2), 2)
    ]
    assert d2["frames_total"] == 2
    assert d2["next_offset"] == len(chunk1) + len(f2[4:])

    # The source database is not affected by the import.
    assert src.get(f"/sessions/{sid}").json()["received_offset"] == len(chunk1)


def test_import_fresh_session_checkpoint(pair):
    src, dst = pair
    register(src)
    register(dst)
    sid = make_session(src, audit=True)["session_id"]

    r = import_cp(dst, export(src, sid))
    assert r.status_code == 201, r.text
    progress = r.json()
    assert progress["received_offset"] == 0
    assert progress["frames_total"] == 0
    assert progress["last_chunk_offset"] is None
    assert progress["sequence_audit"]["bit_width"] == 8
    assert progress["sequence_audit"]["last_sequence"] is None

    # Collection starts from scratch under the same session id.
    d = append(dst, sid, 0, SYNC + build_logical(1, 3, 0)).json()
    assert d["frames_total"] == 1
    assert d["sequence_audit"]["last_sequence"] == 1


def test_imported_audit_session_continues_from_baseline(pair):
    src, dst = pair
    register(src)
    register(dst)
    sid = make_session(src, audit=True)["session_id"]
    f1 = SYNC + build_logical(1, 3, 0)
    f3 = SYNC + build_logical(3, 3, 0)  # seq 2 missing -> suspected gap
    f4 = SYNC + build_logical(4, 3, 0)
    chunk1 = f1 + f3 + f4[:4]
    d1 = append(src, sid, 0, chunk1).json()
    assert d1["sequence_audit"]["suspected_gaps"] == 1

    r = import_cp(dst, export(src, sid))
    assert r.status_code == 201, r.text

    # The replayed response carries the original audit block verbatim.
    replay = append(dst, sid, 0, chunk1).json()
    assert replay["replayed"] is True
    assert replay["sequence_audit"] == d1["sequence_audit"]

    # The split frame (seq 4) completes on the target and seq 6 skips one:
    # the imported baseline keeps judging across the database switch.
    d2 = append(dst, sid, len(chunk1), f4[4:] + SYNC + build_logical(6, 3, 0)).json()
    audit = d2["sequence_audit"]
    assert audit["last_sequence"] == 6
    assert audit["suspected_gaps"] == 2
    assert audit["missing_total"] == 2
    assert [a["sequence"] for a in audit["anomalies"]] == [6]
    assert dst.get(f"/sessions/{sid}").json()["sequence_audit"]["missing_total"] == 2


def test_export_import_export_is_stable(pair):
    src, dst = pair
    _, _, _, cp1 = mid_stream_checkpoint(src)
    register(dst)
    assert import_cp(dst, cp1).status_code == 201
    assert export(dst, cp1["session_id"]) == cp1


# ------------------------------------------------------- import rejections ---


def test_import_same_session_twice_conflicts_without_overwrite(pair):
    src, dst = pair
    sid, chunk, _, cp = mid_stream_checkpoint(src)
    register(dst)

    assert import_cp(dst, cp).status_code == 201
    before = dst.get(f"/sessions/{sid}").json()

    r = import_cp(dst, cp)
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "SESSION_CHECKPOINT_CONFLICT"
    assert err["details"]["reason"] == "session_exists"
    assert err["details"]["session_id"] == sid
    # The already imported session is left untouched.
    assert dst.get(f"/sessions/{sid}").json() == before


def test_import_requires_registered_protocol_version(pair):
    src, dst = pair
    sid, _, _, cp = mid_stream_checkpoint(src)

    r = import_cp(dst, cp)  # target database has no protocols at all
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "PROTOCOL_NOT_FOUND"

    register(dst, version=2)  # only a different version is registered
    r = import_cp(dst, cp)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "PROTOCOL_NOT_FOUND"
    assert dst.get(f"/sessions/{sid}").status_code == 404


def test_import_rejects_same_version_with_different_definition(pair):
    src, dst = pair
    sid, _, _, cp = mid_stream_checkpoint(src)
    register(dst, max_frame_length=128)  # same id+version, other definition

    r = import_cp(dst, cp)
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "SESSION_CHECKPOINT_CONFLICT"
    assert err["details"]["reason"] == "protocol_mismatch"
    assert err["details"]["registered_sha256"] != err["details"]["checkpoint_sha256"]
    assert dst.get(f"/sessions/{sid}").status_code == 404


def _invalid_checkpoints(cp):
    """Mutations of one valid checkpoint, each violating one invariant."""
    cases = {}

    def add(name, mutate):
        mutated = {**cp, "last_response": dict(cp["last_response"])}
        if cp["sequence_audit"] is not None:
            mutated["sequence_audit"] = dict(cp["sequence_audit"])
        mutated["last_response"]["sequence_audit"] = dict(
            cp["last_response"]["sequence_audit"]
        )
        mutate(mutated)
        cases[name] = mutated

    add("unsupported_format_version", lambda c: c.update(checkpoint_version=2))
    add("tail_not_base64", lambda c: c.update(tail_b64="@@@not-b64"))
    add(
        "tail_longer_than_stream",
        lambda c: c.update(tail_b64=b64(b"\x00" * (c["received_offset"] + 1))),
    )
    add("partial_last_chunk_triple", lambda c: c.update(last_response=None))
    add(
        "last_chunk_not_below_offset",
        lambda c: c.update(last_chunk_offset=c["received_offset"]),
    )
    add(
        "response_next_offset_mismatch",
        lambda c: c["last_response"].update(next_offset=c["received_offset"] + 1),
    )
    add(
        "response_chunk_offset_mismatch",
        lambda c: c["last_response"].update(received_offset=c["last_chunk_offset"] + 1),
    )
    add(
        "response_frames_total_mismatch",
        lambda c: c["last_response"].update(frames_total=c["frames_total"] + 1),
    )
    add(
        "response_tail_mismatch",
        lambda c: c["last_response"]["tail"].update(payload_b64=b64(b"\x00")),
    )
    add(
        "response_is_a_replay",
        lambda c: c["last_response"].update(replayed=True),
    )
    add(
        "audit_bit_width_mismatch",
        lambda c: c["sequence_audit"].update(bit_width=4),
    )
    add(
        "audit_sequence_out_of_modulus",
        lambda c: c["sequence_audit"].update(last_sequence=256),
    )
    add(
        "audit_counter_negative",
        lambda c: c["sequence_audit"].update(missing_total=-1),
    )
    add(
        "audit_missing_below_gaps",
        lambda c: c["sequence_audit"].update(missing_total=0, suspected_gaps=1),
    )
    add("audit_dropped", lambda c: c.update(sequence_audit=None))
    add(
        "response_audit_dropped",
        lambda c: c["last_response"].update(sequence_audit=None),
    )
    add(
        "response_audit_counter_mismatch",
        lambda c: c["last_response"]["sequence_audit"].update(duplicates=9),
    )
    return cases


def test_invalid_checkpoints_rejected_without_state_change(pair):
    src, dst = pair
    sid, _, _, cp = mid_stream_checkpoint(src)
    register(dst)

    # The unmodified checkpoint is accepted by the target database.
    assert import_cp(dst, cp).status_code == 201

    for name, mutated in _invalid_checkpoints(cp).items():
        mutated["session_id"] = f"{sid}-{name}"  # avoid the session_exists path
        r = import_cp(dst, mutated)
        assert r.status_code == 422, f"{name}: {r.status_code} {r.text}"
        err = r.json()["error"]
        assert err["code"] == "SESSION_CHECKPOINT_INVALID", name
        assert err["details"]["check"], name
        # Nothing was imported for the rejected checkpoint.
        assert dst.get(f"/sessions/{mutated['session_id']}").status_code == 404, name


def test_non_pristine_checkpoint_without_last_chunk_rejected(pair):
    src, dst = pair
    register(src)
    register(dst)
    sid = make_session(src)["session_id"]
    cp = export(src, sid)  # fresh session: no chunk committed yet

    cp["received_offset"] = 5
    r = import_cp(dst, cp)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SESSION_CHECKPOINT_INVALID"
    assert dst.get(f"/sessions/{sid}").status_code == 404


def test_audit_checkpoint_on_non_audit_protocol_rejected(pair):
    src, dst = pair
    sid, _, _, cp = mid_stream_checkpoint(src, audit=True)
    # Target registers a definition whose sequence field is signed: the
    # digest differs, so this is caught as a protocol mismatch...
    register(dst, fields=[{"name": "seq", "type": "s8", "offset": 0}])
    r = import_cp(dst, cp)
    assert r.status_code == 409
    assert r.json()["error"]["details"]["reason"] == "protocol_mismatch"

    # ...and a checkpoint that claims audit state for the actually
    # registered (audit-incapable) protocol is invalid on its own.
    other = export(src, make_session(src, audit=True)["session_id"])
    register(src, protocol_id="noaudit-line", sequence_field=None)
    digest = register(dst, protocol_id="noaudit-line", sequence_field=None)[
        "definition_sha256"
    ]
    other["protocol_id"] = "noaudit-line"
    other["definition_sha256"] = digest
    r = import_cp(dst, other)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SESSION_CHECKPOINT_INVALID"
    assert r.json()["error"]["details"]["check"] == "sequence_audit"


# ------------------------------------------------------- wide audit values ---


def test_wide_audit_checkpoint_roundtrip_stays_exact(pair):
    src, dst = pair
    for client in (src, dst):
        r = client.post("/protocols", json=u64_protocol())
        assert r.status_code == 201, r.text
    r = src.post(
        "/sessions",
        json={"protocol_id": "sensor-line64", "version": 1, "sequence_audit": True},
    )
    sid = r.json()["session_id"]

    # Confirm 0, 2**63-1, 0, 2**63-1: missing_total reaches 2**64 - 4, past
    # the signed 64-bit range SQLite INTEGER can hold.
    chunk = frame64(0) + frame64(INT64_MAX) + frame64(0) + frame64(INT64_MAX)
    d1 = append(src, sid, 0, chunk).json()
    assert d1["sequence_audit"]["missing_total"] == (1 << 64) - 4
    assert d1["sequence_audit"]["last_sequence"] == INT64_MAX

    # The export JSON carries the exact decimal text, no float rounding.
    raw = src.get(f"/sessions/{sid}/checkpoint")
    assert raw.status_code == 200
    assert str((1 << 64) - 4) in raw.text
    cp = raw.json()
    assert cp["sequence_audit"]["missing_total"] == (1 << 64) - 4

    r = import_cp(dst, cp)
    assert r.status_code == 201, r.text
    audit = r.json()["sequence_audit"]
    assert audit["missing_total"] == (1 << 64) - 4
    assert audit["last_sequence"] == INT64_MAX

    # Resending the last chunk replays the original wide audit block.
    replay = append(dst, sid, 0, chunk).json()
    assert replay["replayed"] is True
    assert replay["sequence_audit"]["missing_total"] == (1 << 64) - 4
    assert dst.get(f"/sessions/{sid}").json()["sequence_audit"]["missing_total"] == (1 << 64) - 4

    # The imported baseline keeps judging exactly beyond 2**63.
    d2 = append(dst, sid, len(chunk), frame64(1 << 63)).json()
    assert d2["sequence_audit"]["anomalies"] == []
    assert d2["sequence_audit"]["last_sequence"] == 1 << 63


# -------------------------------------------------------------- concurrency ---


def test_concurrent_imports_of_same_checkpoint_exactly_one_commits(pair):
    src, dst = pair
    sid, _, _, cp = mid_stream_checkpoint(src)
    register(dst)

    barrier = threading.Barrier(2)

    def race(_):
        barrier.wait()
        return import_cp(dst, cp).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(race, [0, 1]))
    assert sorted(statuses) == [201, 409]
    progress = dst.get(f"/sessions/{sid}").json()
    assert progress["frames_total"] == 1
    assert progress["received_offset"] == cp["received_offset"]
