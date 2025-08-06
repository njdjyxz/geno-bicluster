from __future__ import annotations

import numpy as np
from numba import njit
from scipy.stats import hypergeom


def run_analysis(
    bim_file: str,
    interaction_file: str,
    out_dir: str,
    chrom1: str,
    chrom2: str,
    max_inter_length: int,
    pval_cutoff: float,
    min_int: int
) -> None:
    """End-to-end pipeline: initialise matrices, compute p-values,
    trim intervals, write CSVs."""
    N, n, sig_int, upper, _, _ = initialize_matrices2(
        bim_file, interaction_file, chrom1, chrom2, pval_cutoff
    )

    if n == 0:
        pval_results = {}
    else:
        pval_results = process_significant_interactions(sig_int, max_inter_length, N, n, upper, min_int)

    compute_interval_pval(pval_results, N, n, out_dir, chrom1, chrom2)
    process_pval_file(out_dir, chrom1, chrom2)  # small tweak below


def parse_plink_bim_file(bim_file: str, chrom1: str, chrom2: str):
    # Check file name formate
    file_split = bim_file.split(".")
    if len(file_split)<=1 or file_split[-1] != "bim":
        raise ValueError("Given file is not in Plink Bim file formate, refer to https://www.cog-genomics.org/plink/1.9/formats#bim")


    chrom1_snps = []
    chrom2_snps = []
    
    with open(bim_file) as csv_file:
        line_counter = 1
        previous_position_1 = -float("inf")
        previous_position_2 = -float("inf")

        for line in csv_file:
            row = line.split()
            if len(row) < 4: raise ValueError("Line too short in .bim, refer to https://www.cog-genomics.org/plink/1.9/formats#bim")
            chrom = row[0]
            snp = row[1]

            try:
                position = int(row[3])
            except:
                raise ValueError("Given file is not in Plink Bim file formate, column 3 should be numerical, refer to https://www.cog-genomics.org/plink/1.9/formats#bim")

            # Check the first entry is a chromosome
            if chrom.isnumeric():
                if int(chrom) <0 or int(chrom) >=23:
                    raise ValueError(f"Unknown Chromosome identified at line {line_counter}")
            elif not chrom in ['X','Y','XY','MT']:
                raise ValueError(f"Unknown Chromosome identified at line {line_counter}")
            
            if chrom1 == chrom:
                # Check bim file is sorted by position
                if position < previous_position_1:
                    raise ValueError("bim file should be sorted by 'Position' column")
                previous_position_1 = position

                chrom1_snps.append(snp)

            if chrom2 == chrom:
                # Check bim file is sorted by position
                if position < previous_position_2:
                    raise ValueError("bim file should be sorted by 'Position' column")
                previous_position_2 = position

                chrom2_snps.append(snp)
            
            line_counter +=1
    
    # Checking duplicate SNPs
    chrom1_set = set(chrom1_snps)
    chrom2_set = set(chrom2_snps)

    if len(chrom1_set) != len(chrom1_snps):
        raise ValueError("Duplicate SNPs detected on requested chromosome")
    if len(chrom2_set) != len(chrom2_snps):
        raise ValueError("Duplicate SNPs detected on requested chromosome")
    
    return chrom1_snps, chrom2_snps

def parse_plink_interaction_data(interaction_file: str, pval_cutoff: float, chrom1: str, chrom2: str):
    # Check file name formate
    file_split = interaction_file.split(".")
    if len(file_split)<=2 or file_split[-2:] != ["epi", "qt"]:
        raise ValueError("Given file is not in Plink interaction file formate, refer to https://www.cog-genomics.org/plink/1.9/formats#epi")
    interactions = dict()
    
    with open(interaction_file) as csv_file:
        header = True
        for line in csv_file:
            if header:
                header = False
                continue

            fields = line.split()
            if len(fields)<6:
                raise ValueError("Given file is not in Plink interaction file formate, the file should have at least 6 columns, refer to https://www.cog-genomics.org/plink/1.9/formats#epi")
            chrom_1 = fields[0]  # CHR1
            snp_1 = fields[1]    # SNP1
            chrom_2 = fields[2]  # CHR2
            snp_2 = fields[3]    # SNP2

            try:
                p_value = float(fields[-1])  # P-value
            except:
                raise ValueError("Given file is not in Plink interaction file formate, the last column should be numerical, refer to https://www.cog-genomics.org/plink/1.9/formats#epi")

            if (p_value < pval_cutoff and
                ((chrom_1 == chrom1 and chrom_2 == chrom2) or 
                 (chrom_2 == chrom1 and chrom_1 == chrom2))):
                    
                if (snp_1, snp_2) in interactions:
                    if p_value < interactions[(snp_1, snp_2)]:
                        interactions[(snp_1, snp_2, 0)] = p_value
                elif (snp_2, snp_1) in interactions:
                    if p_value < interactions[(snp_2, snp_1)]:
                        interactions[(snp_2, snp_1, 1)] = p_value
                else:
                    interactions[(snp_1, snp_2, 0)] = p_value

    int_list = []
    for (snp_1, snp_2, r), p_val in interactions.items():
        int_list.append([snp_1, snp_2, r, p_val])
            
    
    return interactions, int_list


def initialize_matrices2(bim_file: str, interaction_files: str, chrom1: str, chrom2: str, 
                         pval_cutoff: float):

    chrom1_snps, chrom2_snps = parse_plink_bim_file(bim_file, chrom1, chrom2)

    interactions, int_list = parse_plink_interaction_data(interaction_files, pval_cutoff, chrom1, chrom2)
    upper_triangular = chrom1 == chrom2
    

    N = 0
    n = 0

    total_markers1 = len(chrom1_snps)
    total_markers2 = len(chrom2_snps)

    if upper_triangular:
        N = sum(range(total_markers1))  # Sum of first total_markers1-1 integers
    else:
        N = total_markers1 * total_markers2

    significant_interactions = np.zeros((len(int_list), 2), dtype=np.int64) # TODO: need to monitor the array size

    idx = 0
    
    snp1_dict = {snp:i for i, snp in enumerate(chrom1_snps)}
    snp2_dict = {snp:j for j, snp in enumerate(chrom2_snps)}
    
    for row in int_list:
        if int(row[2]) == 0:
            significant_interactions[idx, 0] = snp1_dict[row[0]]
            significant_interactions[idx, 1] = snp2_dict[row[1]]
            idx += 1
            n += 1
        else:
            significant_interactions[idx, 0] = snp1_dict[row[1]]
            significant_interactions[idx, 1] = snp2_dict[row[0]]
            idx += 1
            n += 1


    significant_interactions = significant_interactions[:idx]

    print("Interaction data initialized.")
    print("N: ", N)
    print("n: ", n)
    print("Chromosome1: ", chrom1)
    print("Chromosome2: ", chrom2)
    print("P-Value Cutoff: ", pval_cutoff)

    return N, n, significant_interactions, upper_triangular, chrom1_snps, chrom2_snps


@njit
def comp_alt(i: int, j: int, a: int, b: int, significant_interactions: np.ndarray, upper_triangular: bool, compute_k=True):
    k_val = 0
    m_val = 0

    if not upper_triangular:
        m_val = a * b
    else:
        for i2 in range(i, i + a):
            if j < (i2 + 1):
                m_val += max(j + b - (i2 + 1), 0)
            else:
                m_val += b

    if not compute_k:
        return 0, m_val

    for idx in range(significant_interactions.shape[0]):
        i2, j2 = significant_interactions[idx]
        if i <= i2 < i + a and j <= j2 < j + b:
            k_val += 1

    return k_val, m_val



@njit
def compute_k_m_parallel(i_start: int, i_end: int, j_start: int, j_end: int, significant_interactions: np.ndarray, 
                         upper_triangular: bool, N: int, n: int, max_inter_length: int, min_int: int):
    
    pval_results = {}

    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            k_is_zero = False
            a_thresh = b_thresh = 0
            
            for a in range(max_inter_length, 0, -1):
                for b in range(max_inter_length, 0, -1):
                    if i + a <= i_end and j + b <= j_end:
                        
                        if k_is_zero and a <= a_thresh and b <= b_thresh:
                            k, m = comp_alt(i, j, a, b, significant_interactions, upper_triangular, compute_k=False)
                        else:
                            k, m = comp_alt(i, j, a, b, significant_interactions, upper_triangular, compute_k=True)
                            
                            if k <= min_int:
                                k_is_zero = True
                                a_thresh = a
                                b_thresh = b
                        
                        if k > (m * n / N):
                            pval_results[(i, j, a, b)] = (k, m)
                        


    return pval_results


def process_significant_interactions(significant_interactions, max_inter_length, N, n, upper_triangular, min_int):
    results = {}
    
    # Track merged submatrices
    merged_submatrices = []

    for (i, j) in significant_interactions:
        i_start = max(i - max_inter_length + 1, 0)
        i_end = i + max_inter_length
        j_start = max(j - max_inter_length + 1, 0)
        j_end = j + max_inter_length

        merged = False
        # Check for overlaps with existing submatrices
        for idx, (mi_start, mi_end, mj_start, mj_end) in enumerate(merged_submatrices):
            if not (i_end < mi_start or i_start > mi_end or j_end < mj_start or j_start > mj_end):
                # If overlapping, merge the submatrices
                merged_submatrices[idx] = (
                    min(mi_start, i_start),
                    max(mi_end, i_end),
                    min(mj_start, j_start),
                    max(mj_end, j_end)
                )
                merged = True
                break
        
        if not merged:
            merged_submatrices.append((i_start, i_end, j_start, j_end))

    # Process each merged submatrix
    for (mi_start, mi_end, mj_start, mj_end) in merged_submatrices:
        chunk_result = compute_k_m_parallel(mi_start, mi_end, mj_start, mj_end, 
                                            significant_interactions, upper_triangular, N, n, max_inter_length, min_int)
        
        print(f"i_start: {mi_start}; i_end: {mi_end}; j_start: {mj_start}; j_end: {mj_end}. Length: {len(chunk_result)}")

        results.update(chunk_result)
    
    return results



def compute_interval_pval(pval_results: dict, N: int, n: int, out_dir: str, chrom1: str, chrom2: str):
    correction = 0.05 / ((N ** 2) * 0.5)
    computed_pvals = {}

    if chrom1 == chrom2:
        filename = out_dir + f'{chrom1}_pval_results.csv'
    else:
        filename = out_dir + f'chr{chrom1}_chr{chrom2}_pval_results.csv'
    
    with open(filename, 'w') as writer:
        writer.write("i,j,a,b,k,m,p_value\n")
        for key, val in pval_results.items():
            k, m = val

            if (k, m) not in computed_pvals:
                p_value = hypergeom.sf(k - 1, N, n, m)
                computed_pvals[(k, m)] = p_value
            else:
                p_value = computed_pvals[(k, m)]

            if p_value < correction:
                writer.write(f"{key[0]},{key[1]},{key[2]},{key[3]},{k},{m},{p_value}\n")
    
    print(f"Computed p-values and saved to file as {filename}.")
    return pval_results



@njit
def check_overlap(i1, j1, a1, b1, i2, j2, a2, b2):
    if i1 == i2 and j1 == j2 and a1 == a2 and b1 == b2:
        return False

    i1_end = i1 + a1 - 1
    j1_end = j1 + b1 - 1
    i2_end = i2 + a2 - 1
    j2_end = j2 + b2 - 1

    # Check if any of the four regions overlap
    origin_test = (((i1_end) >= i2 >= i1) and ((j1_end) >= j2 >= j1)) or \
                  (((i2_end) >= i1 >= i2) and ((j2_end) >= j1 >= j2))

    bottom_left_test = ((i2 <= i1_end <= i2_end) and (j2_end >= j1 >= j2)) or \
                       ((i1 <= i2_end <= i1_end) and (j1_end >= j2 >= j1))

    bottom_right_test = ((i2 <= i1_end <= i2_end) and (j2_end >= j1_end >= j2)) or \
                        ((i1 <= i2_end <= i1_end) and (j1_end >= j2_end >= j1))

    upper_right_test = (((i1_end) >= i2 >= i1) and ((j2_end) >= j1_end >= j2)) or \
                       (((i2_end) >= i1 >= i2) and ((j1_end) >= j2_end >= j1))

    # Return True if any test detects an overlap
    return origin_test or bottom_left_test or bottom_right_test or upper_right_test

@njit
def trim_intervals(intervals, p_values):
    n = len(intervals)
    keep = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if not keep[i]:
            continue
        i1 = intervals[i, 0]
        j1 = intervals[i, 1]
        a1 = intervals[i, 2]
        b1 = intervals[i, 3]
        pval_i = p_values[i]
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            i2 = intervals[j, 0]
            j2 = intervals[j, 1]
            a2 = intervals[j, 2]
            b2 = intervals[j, 3]
            pval_j = p_values[j]
            if check_overlap(i1, j1, a1, b1, i2, j2, a2, b2):
                if pval_j <= pval_i:
                    keep[i] = False
                    break  # Stop checking further if outer interval is discarded
                else:
                    keep[j] = False
    return keep

def process_pval_file(out_dir, chrom1, chrom2):

    if chrom1 == chrom2:
        input_filename = out_dir + f'{chrom1}_pval_results.csv'
        output_filename = out_dir + f'{chrom1}_pval_results_trimmed.csv'
    else:
        input_filename = out_dir + f'chr{chrom1}_chr{chrom2}_pval_results.csv'
        output_filename = out_dir + f'chr{chrom1}_chr{chrom2}_pval_results_trimmed.csv'

    print(f"Reading p-value results from {input_filename}...", flush=True)

    # Read the CSV file into NumPy arrays
    data = np.genfromtxt(input_filename, delimiter=',', skip_header=1)
    # Adjust indices if needed based on your CSV structure
    i = data[:, 0].astype(np.int64)
    j = data[:, 1].astype(np.int64)
    a = data[:, 2].astype(np.int64)
    b = data[:, 3].astype(np.int64)
    p_values = data[:, 6]

    n = len(i)
    print(f"Total intervals read: {n}", flush=True)

    # Stack intervals into a single array
    intervals = np.stack((i, j, a, b), axis=1)

    # Sort intervals by p-value (ascending order)
    sorted_indices = np.argsort(p_values)
    intervals = intervals[sorted_indices]
    p_values = p_values[sorted_indices]

    print("Trimming overlapping intervals...", flush=True)

    # Call the njit-compiled function to trim intervals
    keep = trim_intervals(intervals, p_values)

    # Select intervals to keep
    trimmed_intervals = intervals[keep]
    trimmed_p_values = p_values[keep]

    print(f"Total intervals after trimming: {len(trimmed_intervals)}", flush=True)

    # Write the trimmed intervals to the output file
    print(f"Writing trimmed intervals to {output_filename}...", flush=True)
    with open(output_filename, 'w') as f_out:
        f_out.write('i,j,a,b,p_value\n')
        for interval, p_value in zip(trimmed_intervals, trimmed_p_values):
            i_out, j_out, a_out, b_out = interval
            f_out.write(f"{i_out},{j_out},{a_out},{b_out},{p_value}\n")

    print(f"File trim completed, trimmed file saved as {output_filename}", flush=True)