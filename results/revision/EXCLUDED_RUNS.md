# Excluded diagnostic output

The files `baselines.jsonl` and `real_baselines.jsonl` contain interrupted
Brandes-only trials and an overlapping restart during development. They are
**not final results** and may contain duplicated records. Do not aggregate them.
The clean exhaustive-insertion runs are `baselines_v2.jsonl` and
`real_baselines_v2.jsonl`; the audit requires unique keys and exact row counts.

`real_rl.jsonl` is an interrupted CPU feasibility measurement, not the complete
real-network quality evaluation. `real_rl_cuda.jsonl` is the complete designated
GPU quality run. No claim compares its latency against a CPU baseline as though
hardware were matched. The original isolated CPU timing experiment is `timing.json`;
the expanded follow-up benchmark below is the current reporting source.

Validation timings collected during training are not deployment benchmarks.
Raw quality-run baseline `seconds` include trajectory-audit calls. They are not
used in the camera-ready timing comparison. The historical `timing.json` measures
all methods with those audit calls excluded and candidate/feature computation
included; `audit_seconds` is retained separately for every method.
Source changes after training affected greedy execution and timing metadata,
not the trained policy computation, reward, graph corpus, or update algorithm.
The source archive and final manifest preserve this distinction.

## Post-review follow-up

The current manuscript uses `followup/timing.jsonl` for its expanded matched
CPU benchmark. The earlier `timing.json` remains a historical measurement,
not a source to pool with the new rows. Incident candidate enumeration was
made direct before the new benchmark.

`followup/real_prefix_raw.jsonl` and `followup/real_tail.jsonl` are executor
partitions retained for provenance. Only their deduplicated merge,
`followup/real.jsonl`, is used as the 300-outcome real quality dataset.
Partitions must not be counted as additional random seeds or observations.
Quality jobs can overlap; their `seconds` fields are diagnostic and are not
used for comparative speedups. The timing stage runs after all quality jobs.
