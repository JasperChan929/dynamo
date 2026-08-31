#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and summarize controlled agent-tree routing benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPEAT_UNIT_CANDIDATES = (" x", " a", ".", "0", "x", "\n")
CONTROL_MESSAGES = [
    {"role": "system", "content": "Agent-tree dispatch control."},
    {"role": "user", "content": "Reply OK."},
]


@dataclass(frozen=True)
class RequestObservation:
    """One sibling request reconstructed from an AIPerf raw record."""

    child_id: str
    request_start_ns: int
    request_end_ns: int
    ttft_ms: float | None
    session_id: str | None
    prefill_worker_id: str | int | None
    decode_worker_id: str | int | None
    prefill_dp_rank: int | None
    prompt_tokens: int | None
    cached_tokens: int | None
    computed_prefill_tokens: int | None
    fully_prefilled_shared_prefix: bool | None

    @property
    def first_hop(self) -> str | None:
        if self.prefill_worker_id is None:
            return None
        if self.prefill_dp_rank is None:
            return str(self.prefill_worker_id)
        return f"{self.prefill_worker_id}/dp{self.prefill_dp_rank}"


@dataclass(frozen=True)
class ExperimentSummary:
    """Placement and duplicate-prefill decision for one benchmark row."""

    observations: tuple[RequestObservation, ...]
    unique_first_hops: tuple[str, ...]
    duplicate_shared_prefix_prefills: int | None
    send_skew_ms: float
    decision: str
    decision_basis: str


def longest_common_prefix_length(sequences: Iterable[list[int]]) -> int:
    """Return the exact common token prefix length across all sequences."""
    materialized = list(sequences)
    if not materialized:
        raise ValueError("at least one token sequence is required")
    for index, values in enumerate(zip(*materialized, strict=False)):
        if len(set(values)) != 1:
            return index
    return min(len(sequence) for sequence in materialized)


def _normalize_chat_tokens(tokens: Any) -> list[int]:
    if isinstance(tokens, Mapping) and "input_ids" in tokens:
        tokens = tokens["input_ids"]
    if isinstance(tokens, list) and all(isinstance(token, int) for token in tokens):
        return tokens
    if isinstance(tokens, list) and len(tokens) == 1:
        encoding_ids = getattr(tokens[0], "ids", None)
        if isinstance(encoding_ids, list) and all(
            isinstance(token, int) for token in encoding_ids
        ):
            return encoding_ids
    raise TypeError(f"unsupported chat-template token result: {type(tokens).__name__}")


def _render_chat_tokens(tokenizer: Any, content: str) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return _normalize_chat_tokens(tokens)


def build_exact_shared_prefix(
    tokenizer: Any,
    target_tokens: int,
    suffixes: list[str],
    repeat_unit: str | None = None,
) -> tuple[str, str]:
    """Build text whose rendered sibling prompts share exactly target_tokens."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if len(suffixes) < 2:
        raise ValueError("at least two sibling suffixes are required")

    units = (repeat_unit,) if repeat_unit is not None else REPEAT_UNIT_CANDIDATES
    shared_tail = "\nShared sibling context ends here.\nBranch "
    for unit in units:
        unit_tokens = tokenizer.encode(unit, add_special_tokens=False)
        if len(unit_tokens) != 1:
            continue
        if tokenizer.encode(unit * 8, add_special_tokens=False) != unit_tokens * 8:
            continue

        baseline = longest_common_prefix_length(
            _render_chat_tokens(tokenizer, shared_tail + suffix) for suffix in suffixes
        )
        repeat_count = max(0, target_tokens - baseline)
        attempted: set[int] = set()
        for _ in range(8):
            attempted.add(repeat_count)
            prefix = unit * repeat_count + shared_tail
            observed = longest_common_prefix_length(
                _render_chat_tokens(tokenizer, prefix + suffix) for suffix in suffixes
            )
            if observed == target_tokens:
                return prefix, unit
            adjusted = max(0, repeat_count + target_tokens - observed)
            if adjusted == repeat_count:
                break
            repeat_count = adjusted

        for distance in range(1, 17):
            for nearby_count in (repeat_count - distance, repeat_count + distance):
                if nearby_count < 0 or nearby_count in attempted:
                    continue
                prefix = unit * nearby_count + shared_tail
                observed = longest_common_prefix_length(
                    _render_chat_tokens(tokenizer, prefix + suffix)
                    for suffix in suffixes
                )
                if observed == target_tokens:
                    return prefix, unit

    requested = f"repeat unit {repeat_unit!r}" if repeat_unit else "default units"
    raise ValueError(
        f"could not construct an exact {target_tokens}-token prefix with {requested}; "
        "pass --repeat-unit with a tokenizer-stable single-token string"
    )


def write_dag_scenario(
    output: Path,
    tokenizer: Any,
    fan_out: int,
    prefix_tokens: int,
    jitter_ms: float,
    max_tokens: int,
    repeat_unit: str | None,
    cache_block_tokens: int = 16,
) -> dict[str, Any]:
    """Write one short control turn and its sibling conversations."""
    if fan_out not in {2, 4, 8}:
        raise ValueError("fan_out must be one of 2, 4, or 8")
    if jitter_ms < 0:
        raise ValueError("jitter_ms must be non-negative")
    if cache_block_tokens <= 0:
        raise ValueError("cache_block_tokens must be positive")
    suffixes = [f"{child_index}: respond briefly." for child_index in range(fan_out)]
    prefix, selected_unit = build_exact_shared_prefix(
        tokenizer, prefix_tokens, suffixes, repeat_unit
    )
    control_tokens = _normalize_chat_tokens(
        tokenizer.apply_chat_template(
            CONTROL_MESSAGES, tokenize=True, add_generation_prompt=True
        )
    )
    first_child_tokens = _render_chat_tokens(tokenizer, prefix + suffixes[0])
    control_overlap = longest_common_prefix_length([control_tokens, first_child_tokens])
    if control_overlap >= cache_block_tokens:
        raise ValueError(
            "control request overlaps a full cache block with the siblings; "
            "select a different tokenizer or cache block size"
        )
    children = [f"child-{child_index}" for child_index in range(fan_out)]
    rows: list[dict[str, Any]] = [
        {
            "session_id": "agent-tree-root",
            "turns": [
                {
                    "messages": CONTROL_MESSAGES,
                    "max_tokens": 1,
                    "spawns": children,
                    "extra": {"temperature": 0.0},
                }
            ],
        }
    ]
    for child_index, (child_id, suffix) in enumerate(
        zip(children, suffixes, strict=True)
    ):
        rows.append(
            {
                "session_id": child_id,
                "turns": [
                    {
                        "messages": [{"role": "user", "content": prefix + suffix}],
                        "max_tokens": max_tokens,
                        "delay": child_index * jitter_ms,
                        "extra": {
                            "temperature": 0.0,
                            "stream_options": {"include_usage": True},
                            "nvext": {"extra_fields": ["worker_id"]},
                        },
                    }
                ],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    return {
        "fan_out": fan_out,
        "shared_prefix_tokens": prefix_tokens,
        "jitter_ms": jitter_ms,
        "repeat_unit": selected_unit,
        "control_overlap_tokens": control_overlap,
        "cache_block_tokens": cache_block_tokens,
        "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "output": str(output),
    }


def _iter_response_chunks(
    record: dict[str, Any],
) -> Iterable[tuple[int, dict[str, Any]]]:
    for response in record.get("responses", []):
        perf_ns = response.get("perf_ns")
        if not isinstance(perf_ns, int):
            continue
        packets = response.get("packets")
        if isinstance(packets, list):
            for packet in packets:
                if packet.get("name") != "data":
                    continue
                value = packet.get("value")
                if not isinstance(value, str) or value.strip() in {"", "[DONE]"}:
                    continue
                yield perf_ns, json.loads(value)
            continue
        text = response.get("text")
        if isinstance(text, str) and text.lstrip().startswith("{"):
            yield perf_ns, json.loads(text)


def _chunk_has_output(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if isinstance(choice.get("text"), str) and choice["text"]:
            return True
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                return True
    return False


def _header(headers: Any, name: str) -> str | None:
    if not isinstance(headers, dict):
        return None
    for key, value in headers.items():
        if key.lower() == name.lower() and isinstance(value, str):
            return value
    return None


def _parse_observation(
    record: dict[str, Any], shared_prefix_tokens: int
) -> RequestObservation:
    metadata = record["metadata"]
    child_id = metadata.get("conversation_id")
    if not isinstance(child_id, str):
        raise TypeError("child record is missing metadata.conversation_id")
    start_perf_ns = record.get("start_perf_ns")
    if not isinstance(start_perf_ns, int):
        raise TypeError(f"{child_id} is missing start_perf_ns")

    first_event_ns: int | None = None
    first_token_ns: int | None = None
    worker_data: dict[str, Any] = {}
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    for perf_ns, chunk in _iter_response_chunks(record):
        if first_event_ns is None:
            first_event_ns = perf_ns
        if first_token_ns is None and _chunk_has_output(chunk):
            first_token_ns = perf_ns
        nvext = chunk.get("nvext")
        if isinstance(nvext, dict) and isinstance(nvext.get("worker_id"), dict):
            worker_data.update(nvext["worker_id"])
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                prompt_tokens = usage["prompt_tokens"]
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict) and isinstance(
                details.get("cached_tokens"), int
            ):
                cached_tokens = details["cached_tokens"]

    ttft_ns = first_token_ns if first_token_ns is not None else first_event_ns
    ttft_ms = None if ttft_ns is None else (ttft_ns - start_perf_ns) / 1_000_000
    computed_prefill_tokens = None
    fully_prefilled = None
    if prompt_tokens is not None and cached_tokens is not None:
        if not 0 <= cached_tokens <= prompt_tokens:
            raise ValueError(f"{child_id} reported invalid token usage")
        computed_prefill_tokens = prompt_tokens - cached_tokens
        fully_prefilled = computed_prefill_tokens >= shared_prefix_tokens

    return RequestObservation(
        child_id=child_id,
        request_start_ns=metadata["request_start_ns"],
        request_end_ns=metadata["request_end_ns"],
        ttft_ms=ttft_ms,
        session_id=_header(record.get("request_headers"), "X-Dynamo-Session-ID"),
        prefill_worker_id=worker_data.get("prefill_worker_id"),
        decode_worker_id=worker_data.get("decode_worker_id"),
        prefill_dp_rank=worker_data.get("prefill_dp_rank"),
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        computed_prefill_tokens=computed_prefill_tokens,
        fully_prefilled_shared_prefix=fully_prefilled,
    )


def _decide(
    baseline: str, unique_first_hops: tuple[str, ...], duplicate_count: int | None
) -> tuple[str, str]:
    if baseline == "B0":
        return "KEEP", "B0 is the placement control; compare the matching B2 row."
    if not unique_first_hops:
        return "KEEP", "Dynamo backend first-hop placement was not exposed."
    if duplicate_count is None:
        return "KEEP", "Server token usage did not expose enough data to count prefill."
    if baseline == "B1":
        if len(unique_first_hops) != 1:
            return (
                "KEEP",
                "The B1 same-worker precondition failed; check session affinity.",
            )
        if duplicate_count > 0:
            return (
                "NEED_C2",
                "Co-located siblings repeatedly computed the shared prefix.",
            )
        return (
            "STOP",
            "B1 co-located siblings and only one full shared prefix was computed.",
        )
    if len(unique_first_hops) > 1 and duplicate_count > 0:
        return (
            "NEED_C1",
            "KV-aware routing spread siblings and duplicated shared-prefix prefill.",
        )
    if len(unique_first_hops) == 1 and duplicate_count > 0:
        return (
            "NEED_C2",
            "Siblings shared one first hop but duplicated shared-prefix prefill.",
        )
    return "STOP", "No cross-worker duplicate shared-prefix prefill was observed."


def summarize_raw_export(
    raw_path: Path,
    baseline: str,
    fan_out: int,
    shared_prefix_tokens: int,
    child_prefix: str = "child-",
) -> ExperimentSummary:
    """Summarize one AIPerf raw export into the C1/C2 decision table."""
    records: list[dict[str, Any]] = []
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                conversation_id = record.get("metadata", {}).get("conversation_id")
                if isinstance(conversation_id, str) and conversation_id.startswith(
                    child_prefix
                ):
                    if record.get("error") is not None or record.get("status") != 200:
                        raise ValueError(
                            f"{conversation_id} did not complete successfully"
                        )
                    records.append(record)
    if len(records) != fan_out:
        raise ValueError(f"expected {fan_out} child records, found {len(records)}")

    observations = tuple(
        sorted(
            (_parse_observation(record, shared_prefix_tokens) for record in records),
            key=lambda observation: observation.request_start_ns,
        )
    )
    first_hops = tuple(
        sorted(
            {
                observation.first_hop
                for observation in observations
                if observation.first_hop is not None
            }
        )
    )
    flags = [observation.fully_prefilled_shared_prefix for observation in observations]
    duplicate_count = None
    if all(isinstance(flag, bool) for flag in flags):
        duplicate_count = max(0, sum(bool(flag) for flag in flags) - 1)
    starts = [observation.request_start_ns for observation in observations]
    send_skew_ms = (max(starts) - min(starts)) / 1_000_000
    decision, decision_basis = _decide(baseline, first_hops, duplicate_count)
    return ExperimentSummary(
        observations=observations,
        unique_first_hops=first_hops,
        duplicate_shared_prefix_prefills=duplicate_count,
        send_skew_ms=send_skew_ms,
        decision=decision,
        decision_basis=decision_basis,
    )


def _display(value: Any, digits: int | None = None) -> str:
    if value is None:
        return "TBD"
    if digits is not None and isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(
    summary: ExperimentSummary,
    csv_path: Path,
    report_path: Path,
    baseline: str,
    fan_out: int,
    shared_prefix_tokens: int,
    jitter_ms: float,
    load: str,
) -> None:
    """Write request-level CSV and the standard benchmark Markdown report."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "child_id",
            "send_offset_ms",
            "ttft_ms",
            "session_id",
            "first_hop",
            "prefill_worker_id",
            "decode_worker_id",
            "prefill_dp_rank",
            "prompt_tokens",
            "cached_tokens",
            "computed_prefill_tokens",
            "fully_prefilled_shared_prefix",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        first_start = min(item.request_start_ns for item in summary.observations)
        for item in summary.observations:
            writer.writerow(
                {
                    "child_id": item.child_id,
                    "send_offset_ms": _display(
                        (item.request_start_ns - first_start) / 1_000_000, 3
                    ),
                    "ttft_ms": _display(item.ttft_ms, 3),
                    "session_id": _display(item.session_id),
                    "first_hop": _display(item.first_hop),
                    "prefill_worker_id": _display(item.prefill_worker_id),
                    "decode_worker_id": _display(item.decode_worker_id),
                    "prefill_dp_rank": _display(item.prefill_dp_rank),
                    "prompt_tokens": _display(item.prompt_tokens),
                    "cached_tokens": _display(item.cached_tokens),
                    "computed_prefill_tokens": _display(item.computed_prefill_tokens),
                    "fully_prefilled_shared_prefix": _display(
                        item.fully_prefilled_shared_prefix
                    ),
                }
            )

    rows = [
        "| Child | First hop | Prompt | Cached | Computed prefill | TTFT ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in summary.observations:
        rows.append(
            "| "
            + " | ".join(
                [
                    item.child_id,
                    _display(item.first_hop),
                    _display(item.prompt_tokens),
                    _display(item.cached_tokens),
                    _display(item.computed_prefill_tokens),
                    _display(item.ttft_ms, 3),
                ]
            )
            + " |"
        )
    arrival = "burst" if jitter_ms == 0 else f"jitter={jitter_ms:g}ms"
    report = [
        "# Experiment",
        "",
        "Question: Is sibling prefix waste caused by first-hop placement or same-worker duplicate prefill?",
        "",
        "Hypothesis: Worker placement and server cache usage distinguish C1 from C2.",
        "",
        f"Workload: fan-out={fan_out} / prefix={shared_prefix_tokens} exact tokens / arrival={arrival} / load={load}",
        "",
        f"Baseline: {baseline}",
        "",
        "Result table:",
        "",
        *rows,
        "",
        f"Observed first hops: {', '.join(summary.unique_first_hops) or 'TBD'}",
        "",
        f"Duplicate shared-prefix prefill count: {_display(summary.duplicate_shared_prefix_prefills)}",
        "",
        f"Actual sibling send skew: {summary.send_skew_ms:.3f}ms",
        "",
        f"Decision: {summary.decision}",
        "",
        f"Decision basis: {summary.decision_basis}",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(report))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one DAG scenario")
    generate.add_argument("--tokenizer", required=True)
    generate.add_argument("--fan-out", type=int, choices=(2, 4, 8), default=2)
    generate.add_argument("--prefix-tokens", type=int, default=16384)
    generate.add_argument("--jitter-ms", type=float, default=0.0)
    generate.add_argument("--max-tokens", type=int, default=8)
    generate.add_argument("--cache-block-tokens", type=int, default=16)
    generate.add_argument("--repeat-unit")
    generate.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize", help="summarize AIPerf raw JSONL")
    summarize.add_argument("--raw", type=Path, required=True)
    summarize.add_argument("--baseline", choices=("B0", "B1", "B2"), required=True)
    summarize.add_argument("--fan-out", type=int, choices=(2, 4, 8), required=True)
    summarize.add_argument("--prefix-tokens", type=int, required=True)
    summarize.add_argument("--jitter-ms", type=float, default=0.0)
    summarize.add_argument("--load", default="TBD")
    summarize.add_argument("--csv", type=Path, required=True)
    summarize.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "generate":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        metadata = write_dag_scenario(
            args.output,
            tokenizer,
            args.fan_out,
            args.prefix_tokens,
            args.jitter_ms,
            args.max_tokens,
            args.repeat_unit,
            args.cache_block_tokens,
        )
        print(json.dumps(metadata, indent=2))
        return

    summary = summarize_raw_export(
        args.raw, args.baseline, args.fan_out, args.prefix_tokens
    )
    write_summary(
        summary,
        args.csv,
        args.report,
        args.baseline,
        args.fan_out,
        args.prefix_tokens,
        args.jitter_ms,
        args.load,
    )
    print(f"Decision: {summary.decision}")


if __name__ == "__main__":
    main()
