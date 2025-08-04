#!/usr/bin/env python3
"""
Lightweight CLI front-end.  All real work is delegated to
geno_bicluster.core.run_analysis().
"""
import argparse
import builtins
import logging
import sys
from pathlib import Path
from time import perf_counter

from . import core  # import your heavy code


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geno-bicluster",
        description="Bicluster significant SNP-SNP interactions (plink-style CLI).",
    )

    p.add_argument("-b", "--bim", required=True, type=Path, help=".bim file from plink")
    p.add_argument("-i", "--interactions", required=True, type=Path, help="Plink *.epi.qt file")
    p.add_argument("-o", "--out-dir", default=".", type=Path, help="Output directory (default: current)")
    p.add_argument("-c1", "--chrom1", required=True, type=str, help="First chromosome")
    p.add_argument("-c2", "--chrom2", required=True, type=str, help="Second chromosome")
    p.add_argument("--max-len", type=int, default=60, help="Maximum interval length (default: 60)")
    p.add_argument("--min-int", type=int, default=0, help="Minimum interactions an interval required to be considered as a valid result")
    p.add_argument("-p", "--p-cutoff", type=float, default=1e-2, help="P-value cutoff (default: 1e-2)")
    return p

def setup_logging(log_path: Path, *, console_level: int = logging.INFO) -> None:
    """
    Write every console message to *both* stderr and a rotating log file,
    while letting existing print() calls continue to work.

    Parameters
    ----------
    log_path : pathlib.Path
        File that receives the log.
    console_level : int
        Logging level that is echoed to the terminal.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("geno_bicluster")
    logger.setLevel(logging.DEBUG)          # capture everything
    logger.handlers.clear()                 # avoid duplicates in Jupyter / re-run

    # ---------- file ----------
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # ---------- console ----------
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)

    # ---------- formatter ----------
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    date = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # ---------- transparently funnel print() → logger.info ----------
    _orig_print = builtins.print

    def print_and_log(*args, **kwargs):
        sep   = kwargs.get("sep", " ")
        end   = kwargs.get("end", "\n")
        file  = kwargs.get("file", sys.stdout)
        flush = kwargs.get("flush", False)

        msg = sep.join(str(a) for a in args)
        logger.info(msg)                    # log *without* trailing newline
        _orig_print(*args, sep=sep, end=end,
                    file=file, flush=flush) # normal console behaviour

    builtins.print = print_and_log


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    t0 = perf_counter()

    print("Running geno-bicluster with:", vars(args), file=sys.stderr)

    # Check file exist (Need to varify TODO)
    if not args.bim.is_file():
        raise FileNotFoundError(f"{str(args.bim)} does not exists or is not a file")
    if not args.interactions.is_file():
        raise FileNotFoundError(f"{str(args.interactions)} does not exists or is not a file")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(Path(args.out_dir) / "geno_bicluster.log",
                  console_level= logging.INFO)

    core.run_analysis(
        bim_file=str(args.bim),
        interaction_file=str(args.interactions),
        out_dir=str(args.out_dir) + "/",  # your code expects trailing slash
        chrom1=args.chrom1,
        chrom2=args.chrom2,
        max_inter_length=args.max_len,
        pval_cutoff=args.p_cutoff,
        min_int = args.min_int
    )

    elapsed = perf_counter() - t0
    print(f"[geno-bicluster] Done in {elapsed:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
