import argparse
import builtins
import datetime
import logging
import sys
from pathlib import Path
from time import perf_counter

from . import core  # import your heavy code

# -------------------------------
# Chromosome spec parsing helpers
# -------------------------------

_VALID_SPECIAL_CHROMS = {"X", "Y", "XY", "MT"}
_SPECIAL_CHROM_ORDER = {"X": 23, "Y": 24, "XY": 25, "MT": 26}

# -------------------------------
# Error handling helpers
# -------------------------------

def _classify_pair_exception(exc: BaseException) -> tuple[str, str]:
    """Classify an exception raised while processing a chromosome pair.

    Returns (action, reason) where action is one of:
      - 'skip'            : tolerate, move to next chromosome pair
      - 'fatal'           : stop the whole run (expected fatal)
      - 'fatal_unexpected': stop the whole run and also log traceback
    """
    msg = str(exc)
    msg_l = msg.lower()

    # --- always-fatal problems ---
    if isinstance(exc, KeyboardInterrupt):
        return ("fatal", "Interrupted by user")
    if isinstance(exc, MemoryError):
        return ("fatal", "Out of memory")
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        return ("fatal", "Missing file or permission error")
    if isinstance(exc, OSError):
        return ("fatal", "OS / filesystem error")

    # --- tolerable per-pair problems (skip) ---
    # core.py: parse_plink_bim_file()
    if isinstance(exc, ValueError) and "duplicate snps detected on requested chromosome" in msg_l:
        return ("skip", "Duplicate SNPs detected on requested chromosome")

    # core.py: process_pval_file() can crash when the pval CSV has only a header (no intervals)
    # Typical numpy message: "too many indices for array"
    if isinstance(exc, IndexError) and "too many indices" in msg_l:
        return ("skip", "No intervals to trim (empty p-value results)")

    # --- fatal, known sanity-check ValueErrors from core.py ---
    if isinstance(exc, ValueError):
        # BIM-related sanity checks
        if "given file is not in plink bim file formate" in msg_l:
            return ("fatal", "BIM sanity check failed (not a valid .bim)")
        if "line too short in .bim" in msg_l:
            return ("fatal", "BIM sanity check failed (line too short)")
        if "column 3 should be numerical" in msg_l:
            return ("fatal", "BIM sanity check failed (non-numeric position column)")
        if "unknown chromosome identified at line" in msg_l:
            return ("fatal", "BIM sanity check failed (unknown chromosome code)")
        if "bim file should be sorted by 'position' column" in msg_l:
            return ("fatal", "BIM sanity check failed (file not sorted by position)")

        # Interaction-file-related sanity checks
        if "given file is not in plink interaction file formate" in msg_l:
            return ("fatal", "Interaction file sanity check failed (not a valid .epi.qt)")
        if "at least 6 columns" in msg_l:
            return ("fatal", "Interaction file sanity check failed (too few columns)")
        if "last column should be numerical" in msg_l:
            return ("fatal", "Interaction file sanity check failed (non-numeric p-value column)")

    # BIM/interaction mismatch: initialize_matrices2() can KeyError if interaction SNP not in BIM
    if isinstance(exc, KeyError):
        return ("fatal", "Interaction references SNP not found in BIM (input mismatch)")

    # Anything else: unexpected -> stop and include traceback
    return ("fatal_unexpected", "Unexpected error")



def _chrom_sort_key(chrom: str) -> int:
    c = chrom.upper()
    if c.isdigit():
        return int(c)
    return _SPECIAL_CHROM_ORDER.get(c, 10**9)

def parse_chrom_spec(spec: str) -> list[str]:
    """
    Parse a chromosome specification string.

    Supported:
      - Single chrom token: 1..22, X, Y, XY, MT (case-insensitive)
      - Comma-separated list: "1,2,10,X"
      - Numeric ranges: "1-10" (digits only; letters are NOT allowed in ranges)
      - Mix ranges and tokens: "1-10,X"

    Disallowed:
      - Letter ranges like "1-X" or "X-10"
    """
    if spec is None:
        raise ValueError("Chromosome spec cannot be None")
    raw = spec.strip()
    if not raw:
        raise ValueError("Chromosome spec cannot be empty")

    items = [s.strip() for s in raw.split(",")]
    if any(it == "" for it in items):
        raise ValueError(f"Invalid chromosome spec '{spec}': empty item (check commas)")

    out: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)

    for item in items:
        if "-" in item:
            if item.count("-") != 1:
                raise ValueError(f"Invalid range '{item}' in '{spec}'")
            left, right = (x.strip() for x in item.split("-", 1))

            # enforce: ranges must be numeric-only (your “no 1-X” rule)
            if not left.isdigit() or not right.isdigit():
                raise ValueError(
                    f"Invalid chromosome range '{item}' in '{spec}': ranges must be numeric, "
                    "and letter chromosomes must be provided as separate comma-separated tokens (e.g. '1-10,X')."
                )

            a, b = int(left), int(right)
            if a < 1 or b > 22:
                raise ValueError(f"Invalid numeric range '{item}' in '{spec}': allowed range is 1-22")
            if a > b:
                raise ValueError(f"Invalid numeric range '{item}' in '{spec}': start must be <= end")

            for k in range(a, b + 1):
                _add(str(k))

        else:
            tok = item.strip().upper()

            # allow common "chr" prefix in single tokens, e.g. chr1, chrx
            if tok.startswith("CHR"):
                tok = tok[3:]

            if tok.isdigit():
                v = int(tok)
                if v < 1 or v > 22:
                    raise ValueError(f"Invalid chromosome '{item}' in '{spec}': numeric chrom must be 1-22")
                _add(str(v))
            elif tok in _VALID_SPECIAL_CHROMS:
                _add(tok)
            else:
                raise ValueError(
                    f"Invalid chromosome token '{item}' in '{spec}'. "
                    "Allowed: 1..22, X, Y, XY, MT; plus numeric ranges like 1-10 and comma-separated lists."
                )

    return out

def expand_chrom_pairs(chrom1_spec: str, chrom2_spec: str) -> list[tuple[str, str]]:
    """
    Expand chrom1/chrom2 specs to a sorted list of unique pairs where order doesn't matter.
    Example: 1-3 and 1-3 => (1,1),(1,2),(1,3),(2,2),(2,3),(3,3)
    """
    c1 = parse_chrom_spec(chrom1_spec)
    c2 = parse_chrom_spec(chrom2_spec)

    pairs: set[tuple[str, str]] = set()
    for a in c1:
        for b in c2:
            if _chrom_sort_key(a) <= _chrom_sort_key(b):
                pairs.add((a, b))
            else:
                pairs.add((b, a))

    return sorted(pairs, key=lambda p: (_chrom_sort_key(p[0]), _chrom_sort_key(p[1])))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geno-bicluster",
        description="geno-bicluster is a Numba-accelerated command-line tool that scans genome-wide SNP-SNP interaction tables produced by PLINK's --epistasis module, identifies rectangular 'hot-spot' sub-matrices whose interaction density exceeds hypergeometric expectations, and reports non-overlapping biclusters together with corrected P-values",
    )

    p.add_argument(
        "-b",
        "--bim",
        required=True,
        type=Path,
        help="PLINK marker map (*.bim) that contains BOTH target chromosomes.",
    )
    p.add_argument(
        "-i",
        "--interactions",
        required=True,
        type=Path,
        help="PLINK SNP-SNP interaction table (*.epi.qt) from --epistasis.",
    )
    p.add_argument(
        "-o", "--out-dir", default=".", type=Path, help="Output directory (default: current)"
    )
    p.add_argument(
        "-c1",
        "--chrom1",
        required=True,
        type=str,
        help="First chromosome (e.g. 1-22, X, Y, XY, MT).",
    )
    p.add_argument(
        "-c2",
        "--chrom2",
        required=True,
        type=str,
        help="Second chromosome (may equal --chrom1 for intra-chrom scans).",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=60,
        help="Maximum side length of a candidate bicluster (SNPs) (default: 60)",
    )
    p.add_argument(
        "--min-int",
        type=int,
        default=1,
        help="Skip rectangles that contain ≤ N significant interactions (defualt: 0)",
    )
    p.add_argument(
        "-p",
        "--p-cutoff",
        type=float,
        default=1e-2,
        help="P-value threshold when reading *.epi.qt* (default: 1e-2)",
    )
    return p


def setup_logging(log_path: Path) -> None:
    """
    Write every console message to *both* stderr and a rotating log file,
    while letting existing print() calls continue to work.

    Parameters
    ----------
    log_path : pathlib.Path
        File that receives the log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("geno_bicluster")
    logger.setLevel(logging.DEBUG)  # capture everything
    logger.handlers.clear()  # avoid duplicates in Jupyter / re-run

    # ---------- file ----------
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    logging.getLogger("numba").setLevel(logging.WARNING)

    # ---------- formatter ----------
    fmt = "%(asctime)s|%(message)s"
    date = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    # ---------- transparently funnel print() → logger.info ----------
    _orig_print = builtins.print

    def print_and_log(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        file = kwargs.get("file", sys.stdout)
        flush = kwargs.get("flush", False)

        msg = sep.join(str(a) for a in args)
        logger.info(msg)  # log *without* trailing newline
        _orig_print(*args, sep=sep, end=end, file=file, flush=flush)  # normal console behaviour

    builtins.print = print_and_log


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    t0 = perf_counter()

    print(
        "Running geno-bicluster with:\n",
        "\n".join([f"\t{k}: {v}" for k, v in vars(args).items()]),
        file=sys.stderr,
    )

    if not args.bim.is_file():
        raise FileNotFoundError(f"{str(args.bim)} does not exists or is not a file")
    if not args.interactions.is_file():
        raise FileNotFoundError(f"{str(args.interactions)} does not exists or is not a file")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_name = args.out_dir.resolve().name or "out"
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")

    setup_logging(args.out_dir / f"geno_bicluster_{out_name}_{ts}.log")

    # Expand chromosome specs (supports ranges like "1-10,X") into unique unordered pairs.
    chrom_pairs = expand_chrom_pairs(args.chrom1, args.chrom2)
    print(f"[geno-bicluster] Expanded to {len(chrom_pairs)} chromosome pair(s): {chrom_pairs}", file=sys.stderr)

    ok = 0
    skipped = 0
    failed = 0
    exit_code: int | None = None

    for c1, c2 in chrom_pairs:
        pair_t0 = perf_counter()
        print(f"[geno-bicluster] Running pair: {c1} vs {c2}", file=sys.stderr)

        try:
            # Run the pipeline in smaller steps so we can decide what to do when n==0 or N==0.
            N, n, sig_int, upper, _, _ = core.initialize_matrices2(
                str(args.bim), str(args.interactions), c1, c2, args.p_cutoff
            )

            # If N==0 -> FATAL (your policy)
            if N == 0:
                failed += 1
                print(
                    f"[geno-bicluster] ERROR: stopping because N==0 for pair {c1} vs {c2}. "
                    "This indicates 0 markers for at least one requested chromosome in the BIM selection.",
                    file=sys.stderr,
                )
                exit_code = 2
                break

            # If n==0 (no significant interactions) -> SKIP (your policy)
            if n == 0:
                skipped += 1
                print(
                    f"[geno-bicluster] WARNING: skipping pair {c1} vs {c2} "
                    f"(n==0: no significant interactions at p<{args.p_cutoff}).",
                    file=sys.stderr,
                )
                continue

            pval_results = core.process_significant_interactions(
                sig_int, args.max_len, N, n, upper, args.min_int
            )
            core.compute_interval_pval(pval_results, N, n, str(args.out_dir) + "/", c1, c2)
            core.process_pval_file(str(args.out_dir) + "/", c1, c2)

            ok += 1

        except SystemExit:
            # Don’t swallow explicit exits
            raise

        except BaseException as e:
            action, reason = _classify_pair_exception(e)

            if action == "skip":
                skipped += 1
                print(
                    f"[geno-bicluster] WARNING: skipping pair {c1} vs {c2} ({reason}). "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                continue

            failed += 1
            print(
                f"[geno-bicluster] ERROR: stopping on pair {c1} vs {c2} ({reason}). "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )

            # For unexpected errors, put traceback into the log file
            if action == "fatal_unexpected":
                logging.getLogger("geno_bicluster").exception(
                    "Unexpected exception for pair %s vs %s", c1, c2
                )

            exit_code = 1
            break

        finally:
            pair_elapsed = perf_counter() - pair_t0
            print(
                f"[geno-bicluster] Finished pair: {c1} vs {c2} | elapsed {pair_elapsed:.1f} s",
                file=sys.stderr,
            )

    # optional summary
    elapsed = perf_counter() - t0
    print(
        f"[geno-bicluster] Summary: ok={ok}, skipped={skipped}, failed={failed} | total elapsed {elapsed:.1f} s",
        file=sys.stderr,
    )

    if exit_code is not None:
        raise SystemExit(exit_code)

    elapsed = perf_counter() - t0
    print(f"[geno-bicluster] Done in {elapsed:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
