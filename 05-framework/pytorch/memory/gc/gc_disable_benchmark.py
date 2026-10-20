"""Compare a short-lived, acyclic workload with cyclic GC enabled/disabled.

Run with:
    python gc_disable_benchmark.py

This is a small benchmark, not proof that disabling GC helps every workload.
The workload holds batches of acyclic lists and dictionaries long enough for
automatic cyclic-GC thresholds to be reached. Reference counting remains on.
"""

import argparse
import gc
import statistics
import time


def process_records(count: int, batch_size: int = 2_000) -> int:
    """Build acyclic containers in batches and return a stable checksum."""
    checksum = 0
    for batch_start in range(0, count, batch_size):
        batch_end = min(batch_start + batch_size, count)
        records = []
        for i in range(batch_start, batch_end):
            records.append({
                "id": i,
                "values": [i, i + 1, i + 2],
                "metadata": {"valid": i % 2, "group": i % 7},
            })

        # Keep this batch alive while it is processed. The tracked containers
        # accumulate long enough to cross GC thresholds, then refcounting frees
        # them when the batch goes out of scope. No references point in a cycle.
        for record in records:
            checksum += sum(record["values"])
            checksum += record["metadata"]["valid"] + record["metadata"]["group"]
    return checksum


def collection_count() -> int:
    """Return the total number of cyclic-GC collections across generations."""
    return sum(generation["collections"] for generation in gc.get_stats())


def timed_run(count: int, enabled: bool) -> tuple[float, int, int]:
    """Return elapsed seconds, checksum, and collections during the workload."""
    # Start each run with comparable GC state; setup time is outside the timer.
    gc.collect()
    if enabled:
        gc.enable()
    else:
        gc.disable()

    before_collections = collection_count()
    start = time.perf_counter()
    checksum = process_records(count)
    elapsed = time.perf_counter() - start
    collections = collection_count() - before_collections
    return elapsed, checksum, collections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=300_000,
                        help="records created per run (default: 300000)")
    parser.add_argument("--repeat", type=int, default=5,
                        help="timed runs per mode (default: 5)")
    args = parser.parse_args()
    if args.records <= 0 or args.repeat <= 0:
        parser.error("--records and --repeat must be positive")

    was_enabled = gc.isenabled()
    print(f"Python GC thresholds: {gc.get_threshold()}")
    print(f"Workload: {args.records:,} records per run; {args.repeat} runs per mode")
    print("Objects created are acyclic; reference counting remains active.\n")

    try:
        # Warm up interpreter/code paths before collecting timings.
        process_records(min(args.records, 10_000))

        results: dict[bool, list[tuple[float, int, int]]] = {True: [], False: []}
        # Alternate order to reduce bias from always running one mode first.
        for trial in range(args.repeat):
            modes = (True, False) if trial % 2 == 0 else (False, True)
            for enabled in modes:
                results[enabled].append(timed_run(args.records, enabled))

        enabled_checksums = {result[1] for result in results[True]}
        disabled_checksums = {result[1] for result in results[False]}
        if enabled_checksums != disabled_checksums:
            raise RuntimeError("GC modes produced different checksums")

        enabled_times = [result[0] for result in results[True]]
        disabled_times = [result[0] for result in results[False]]
        enabled_median = statistics.median(enabled_times)
        disabled_median = statistics.median(disabled_times)
        saved_percent = (enabled_median - disabled_median) / enabled_median * 100

        print(f"GC enabled:  median {enabled_median:.4f}s; "
              f"collections per run {[r[2] for r in results[True]]}")
        print(f"GC disabled: median {disabled_median:.4f}s; "
              f"collections per run {[r[2] for r in results[False]]}")
        print(f"Median time change with GC disabled: {saved_percent:+.2f}% "
              "(positive means faster)")
        print(f"Matching checksum: {next(iter(enabled_checksums))}")
        print("Results vary by Python version, machine, and workload.")
    finally:
        # Restore the caller's original setting, even if an error occurs.
        if was_enabled:
            gc.enable()
            gc.collect()
        else:
            gc.disable()


if __name__ == "__main__":
    main()
