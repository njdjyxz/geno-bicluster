# geno-bicluster

**Fast biclustering of PLINK SNP–SNP interaction matrices ( *.epi.qt* )**

`geno-bicluster` pinpoints rectangular “hot-spot” regions in a chromosome-by-chromosome
interaction matrix whose density of significant epistatic hits is unlikely under a
hypergeometric null.  The implementation is Numba-accelerated (~10–50× faster
than naïve Python) and handles both intra- and inter-chromosomal scans.

---

## Quick start

### Install

```bash
pip install "git+https://github.com/njdjyxz/geno-bicluster.git@main"
```

### Command-line reference
```bash
required:
  -b / --bim FILE           PLINK marker map
  -i / --interactions FILE  PLINK --epistasis output (SNP x SNP)
  -c1 / --chrom1 CHR        first chromosome
  -c2 / --chrom2 CHR        second chromosome

optional:
  -p / --p-cutoff FLOAT     P-value cutoff while reading table   [1e-2]
  --max-len INT             longest rectangle side (SNPs)        [60]
  --min-int INT             min # significant hits per rectangle [0]
  -o / --out-dir DIR        output folder                        [.]
```
