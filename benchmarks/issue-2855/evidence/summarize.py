"""Summarize an A5 URMA spot check with independently timed device and host spans."""

import csv
import re
import sys
from collections import defaultdict
from math import ceil
from pathlib import Path
from statistics import mean

from simpler_setup.tools.strace_timing import parse_spans

ROOT = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
ROUNDS = 80
WARMUP = 8
SPANS = {
    "chip.run.runner_run.device_wall.task_slot_0": "device_put_to_consumer",
    "chip.run": "host_native_chip_run",
}
L3 = re.compile(r"^URMA_HOST_L3_NS (\d+)$", re.MULTILINE)


def samples(path):
    log = path.read_text()
    assert re.search(r"^=+ 1 passed in ", log, re.MULTILINE), path
    by_span = defaultdict(lambda: defaultdict(dict))
    for span in parse_spans(log.splitlines()):
        if span.name in SPANS:
            rank_spans = by_span[span.name][span.pid]
            assert span.inv not in rank_spans and span.dur > 0, (path, span)
            rank_spans[span.inv] = span.dur
    for name, metric in SPANS.items():
        ranks = by_span[name]
        assert len(ranks) == 2 and all(len(rank) == ROUNDS for rank in ranks.values()), (
            path, metric, {pid: len(rank) for pid, rank in ranks.items()}
        )
        for rank_idx, rank in enumerate(ranks.values()):
            for ordinal, inv in enumerate(sorted(rank), 1):
                yield metric, rank_idx, ordinal, rank[inv]
    l3 = list(map(int, L3.findall(log)))
    assert len(l3) == ROUNDS and all(l3), (path, len(l3))
    for ordinal, elapsed in enumerate(l3, 1):
        yield "host_l3_worker_run", "", ordinal, elapsed


def summary_row(case, kib, metric, durations):
    durations = sorted(durations)
    ns_per_us = 1000
    pct = [f"{durations[ceil(len(durations) * p) - 1] / ns_per_us:.3f}" for p in (0.50, 0.95, 0.99)]
    gb_s = f"{33 * kib * 1024 / mean(durations):.4f}" if metric == "device_put_to_consumer" else ""
    return [case, kib, metric, len(durations), *pct, gb_s]


with (OUTPUT / "head-summary.csv").open("w", newline="") as summary, (
    OUTPUT / "head-samples.csv"
).open("w", newline="") as raw:
    summaries = csv.writer(summary)
    rows = csv.writer(raw)
    summaries.writerow(["case", "request_kib", "metric", "samples", "p50_us", "p95_us", "p99_us", "per_rank_gb_s"])
    rows.writerow(["case", "metric", "rank_index", "round", "duration_ns", "warmup"])
    paths = sorted(ROOT.glob("put_*k_*.log"))
    assert len(paths) == 10, paths
    for path in paths:
        case = path.stem
        kib = int(case.split("_")[1][:-1])
        metrics = defaultdict(list)
        for metric, rank, ordinal, duration in samples(path):
            rows.writerow([case, metric, rank, ordinal, duration, int(ordinal <= WARMUP)])
            if ordinal > WARMUP:
                metrics[metric].append(duration)
        for metric, durations in metrics.items():
            summaries.writerow(summary_row(case, kib, metric, durations))
