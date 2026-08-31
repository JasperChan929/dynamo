# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.router.agent_tree import (
    build_exact_shared_prefix,
    longest_common_prefix_length,
    summarize_raw_export,
    write_dag_scenario,
)


class CharacterTokenizer:
    """Minimal tokenizer whose token IDs are Unicode code points."""

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        content = messages[0]["content"]
        return self.encode(f"<user>{content}<assistant>")


def _raw_record(child_id, worker_id, prompt_tokens, cached_tokens, session_id):
    first_chunk = {
        "choices": [{"delta": {"content": "x"}}],
        "nvext": {
            "worker_id": {
                "prefill_worker_id": worker_id,
                "decode_worker_id": worker_id,
                "prefill_dp_rank": 0,
            }
        },
    }
    usage_chunk = {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }
    return {
        "metadata": {
            "conversation_id": child_id,
            "request_start_ns": 1_000_000,
            "request_end_ns": 3_000_000,
            "benchmark_phase": "profiling",
        },
        "start_perf_ns": 10_000_000,
        "request_headers": {"X-Dynamo-Session-ID": session_id},
        "status": 200,
        "responses": [
            {
                "perf_ns": 11_000_000,
                "packets": [{"name": "data", "value": json.dumps(first_chunk)}],
            },
            {
                "perf_ns": 12_000_000,
                "packets": [{"name": "data", "value": json.dumps(usage_chunk)}],
            },
        ],
    }


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_generate_scenario_has_exact_shared_token_prefix(tmp_path):
    tokenizer = CharacterTokenizer()
    output = tmp_path / "scenario.jsonl"

    metadata = write_dag_scenario(output, tokenizer, 4, 128, 2.0, 8, None)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    child_contents = [row["turns"][0]["messages"][0]["content"] for row in rows[1:]]
    tokenized = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for content in child_contents
    ]
    assert longest_common_prefix_length(tokenized) == 128
    assert [row["turns"][0]["delay"] for row in rows[1:]] == [
        0.0,
        2.0,
        4.0,
        6.0,
    ]
    assert rows[0]["turns"][0]["spawns"] == [
        "child-0",
        "child-1",
        "child-2",
        "child-3",
    ]
    assert metadata["fan_out"] == 4
    assert metadata["control_overlap_tokens"] < 16


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_exact_prefix_builder_rejects_too_few_suffixes():
    with pytest.raises(ValueError, match="at least two"):
        build_exact_shared_prefix(CharacterTokenizer(), 32, ["only-child"])


@pytest.mark.parametrize(
    "baseline,workers,cached_tokens,expected_decision,expected_duplicates",
    [
        ("B2", ("worker-0", "worker-1"), (0, 0), "NEED_C1", 1),
        ("B1", ("worker-0", "worker-0"), (0, 100), "STOP", 0),
    ],
)
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_summarize_raw_export_attributes_prefill_waste(
    tmp_path,
    baseline,
    workers,
    cached_tokens,
    expected_decision,
    expected_duplicates,
):
    raw_path = tmp_path / "profile_export_raw.jsonl"
    records = [
        _raw_record("child-0", workers[0], 101, cached_tokens[0], "session-0"),
        _raw_record("child-1", workers[1], 101, cached_tokens[1], "session-1"),
    ]
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = summarize_raw_export(raw_path, baseline, 2, 100)

    assert summary.decision == expected_decision
    assert summary.duplicate_shared_prefix_prefills == expected_duplicates
    assert summary.observations[0].ttft_ms == 1.0
