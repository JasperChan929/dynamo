<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent-tree sibling placement benchmark

This benchmark answers a narrow question: when several child requests arrive together with a long
exact shared prefix, is duplicate prefill caused by first-hop placement across Dynamo workers or by
same-worker engine scheduling?

It uses AIPerf's DAG SPAWN path for concurrent dispatch and its raw export for backend attribution.
The `worker_id` in AIPerf's normal records export identifies an AIPerf client worker, not a Dynamo
backend worker, so it must not be used for this diagnosis.

## Requirements

- Install the benchmark dependencies from `benchmarks/pyproject.toml`, including the pinned AIPerf
  0.10.0 and Transformers.
- Run at least two Dynamo workers for B0 and B2. Use separate physical GPUs when drawing a
  cross-GPU conclusion.
- Ask the backend to report cache usage. vLLM requires `--enable-prefix-caching` and
  `--enable-prompt-tokens-details`.
- Keep the model and tokenizer identical. The generator verifies the exact common prefix with the
  selected tokenizer; server `usage.prompt_tokens` remains the runtime sanity check.

## Generate one workload row

The supported matrix is fan-out 2/4/8, shared prefix 16k/32k, and burst or small deterministic
jitter. One generated file represents one matrix row.

```bash
python benchmarks/router/agent_tree.py generate \
  --tokenizer Qwen/Qwen3-0.6B \
  --fan-out 2 \
  --prefix-tokens 16384 \
  --jitter-ms 0 \
  --cache-block-tokens 16 \
  --output /tmp/agent-tree/f2-p16k-burst.dag.jsonl
```

The first DAG node sends one short control request, then SPAWN dispatches complete, independent
child payloads. The generator verifies that the control and child prompts overlap by less than one
cache block, while the rendered child prompts have exactly the requested common prefix. Therefore
the control request cannot warm a full block of the measured sibling prefix.

## Run B0, B1, and B2

Use the same generated workload and cold cache for each row. Do not configure an AIPerf warmup.
`--num-conversations 1` runs the one control root and its complete child fan-out without a
wire-request cap that could truncate the tree.

For B0 default routing and B2 KV-aware routing, do not send a Dynamo session header:

```bash
AIPERF_DAG_FAIL_FAST=1 aiperf profile \
  --url http://localhost:8000 \
  --model Qwen/Qwen3-0.6B \
  --tokenizer Qwen/Qwen3-0.6B \
  --endpoint-type chat \
  --streaming \
  --input-file /tmp/agent-tree/f2-p16k-burst.dag.jsonl \
  --custom-dataset-type dag_jsonl \
  --num-conversations 1 \
  --concurrency 1 \
  --export-level raw \
  --artifact-dir /tmp/agent-tree/b2
```

B0 uses the default router. B2 uses the same client command against a frontend started with
`--router-mode kv`; record the exact event/prediction settings with the result.

For the B1 same-worker control, start the frontend with session affinity and force one shared
session header for the row:

```bash
AIPERF_DAG_FAIL_FAST=1 aiperf profile \
  --url http://localhost:8000 \
  --model Qwen/Qwen3-0.6B \
  --tokenizer Qwen/Qwen3-0.6B \
  --endpoint-type chat \
  --streaming \
  --input-file /tmp/agent-tree/f2-p16k-burst.dag.jsonl \
  --custom-dataset-type dag_jsonl \
  --num-conversations 1 \
  --concurrency 1 \
  --export-level raw \
  --header X-Dynamo-Session-ID:agent-tree-b1-f2-p16k \
  --artifact-dir /tmp/agent-tree/b1
```

Use a new generated prefix or restart/clear the workers between trials. Reusing a warm prefix turns
the cold first-hop experiment into a steady-state cache-hit experiment.

## Build the result table

```bash
python benchmarks/router/agent_tree.py summarize \
  --raw /tmp/agent-tree/b2/profile_export_raw.jsonl \
  --baseline B2 \
  --fan-out 2 \
  --prefix-tokens 16384 \
  --load idle \
  --csv /tmp/agent-tree/b2/requests.csv \
  --report /tmp/agent-tree/b2/report.md
```

The parser reads `nvext.worker_id.prefill_worker_id` and `prefill_dp_rank` from raw SSE chunks. It
derives:

```text
computed_prefill_tokens = prompt_tokens - cached_tokens
duplicate_shared_prefix_prefills = max(0, full_prefix_prefill_requests - 1)
```

Missing server fields remain `TBD`; the parser does not substitute the AIPerf client worker ID.
Interpret decisions as follows:

| Observation | Decision |
|---|---|
| B2 spreads siblings and more than one request computes the full prefix | `NEED_C1` |
| One first hop still computes the full prefix more than once | `NEED_C2` |
| No duplicate full-prefix prefill is observed | `STOP` |
| Placement or cache usage is not exposed, or the B1 co-location control fails | `KEEP` |

Report actual request-start skew even for a configured zero-jitter burst. Keep worker-to-physical-GPU
mapping, load, model revision, block size, cache state, and router configuration with every result.
Do not describe a multi-process single-GPU result as cross-GPU evidence.
