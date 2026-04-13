"""
Phase 2 — Measure query runtimes (warm-up + N timed runs, median, optional timeout).
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from src.utils import (  # noqa: E402
    ensure_data_dirs,
    list_sqlite_databases,
    load_config,
    resolve_bird_root,
    load_jsonl,
    save_jsonl,
)


def measure_runtime(
    db_path: Path,
    query: str,
    n_runs: int,
    cache_size_pages: int,
    timeout_s: float,
) -> tuple[float | None, list[float], bool, str | None]:
    """
    Returns (median_seconds, all_runs, timed_out, error_message).
    On failure before timing: (None, [], False, err).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute(f"PRAGMA cache_size = {cache_size_pages}")
        try:
            conn.execute(query).fetchall()
        except sqlite3.Error as e:
            return None, [], False, str(e)

        times: list[float] = []
        timed_out = False
        for _ in range(n_runs):
            start = time.perf_counter()
            try:
                conn.execute(query).fetchall()
            except sqlite3.Error as e:
                return None, times, timed_out, str(e)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            if elapsed > timeout_s:
                timed_out = True
                break

        if not times:
            return None, [], timed_out, "no successful timed runs"
        med = sorted(times)[len(times) // 2]
        return med, times, timed_out, None
    finally:
        conn.close()


def collect_for_file(
    jsonl_path: Path,
    db_dir: Path,
    n_runs: int,
    cache_size_pages: int,
    timeout_s: float,
) -> list[dict]:
    records = load_jsonl(jsonl_path)
    out = []
    for rec in records:
        db_name = rec["database"]
        db_path = db_dir / db_name / f"{db_name}.sqlite"
        if not db_path.is_file():
            continue
        q = rec["query_text"]
        med, all_t, tout, err = measure_runtime(
            db_path, q, n_runs, cache_size_pages, timeout_s
        )
        if err:
            item = {
                **rec,
                "median_runtime_seconds": None,
                "all_runtimes": [],
                "timed_out": False,
                "error": err,
            }
        else:
            item = {
                **rec,
                "median_runtime_seconds": med,
                "all_runtimes": all_t,
                "timed_out": tout,
                "error": None,
            }
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect runtimes for synthetic queries.")
    parser.add_argument("--input", type=str, default=None, help="Single .jsonl file")
    args = parser.parse_args()

    cfg = load_config()
    dirs = ensure_data_dirs(_PKG, cfg)
    bird_root = resolve_bird_root(cfg)
    db_dir = bird_root / "dev_databases"

    meta = {
        "cpu": platform.processor() or platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
    }
    try:
        import sqlite3 as sq

        meta["sqlite"] = sq.sqlite_version
    except Exception:
        meta["sqlite"] = "unknown"

    n_runs = int(cfg["timing_runs"])
    cache_sz = int(cfg["cache_size_pages"])
    timeout_s = float(cfg["timeout_seconds"])

    synth_dir = dirs["synthetic"]
    out_dir = dirs["runtimes"]
    (out_dir / "collection_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if args.input:
        files = [Path(args.input)]
    else:
        files = sorted(synth_dir.glob("*.jsonl"))

    if not files:
        print(f"No input jsonl in {synth_dir}. Run generate_queries first.")
        sys.exit(1)

    for fp in files:
        print(f"Timing queries from {fp.name} ...")
        timed = collect_for_file(fp, db_dir, n_runs, cache_sz, timeout_s)
        out_path = out_dir / fp.name
        save_jsonl(out_path, timed)
        ok = sum(1 for r in timed if r.get("median_runtime_seconds") is not None)
        print(f"  {ok}/{len(timed)} successful -> {out_path}")


if __name__ == "__main__":
    main()
