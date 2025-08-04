"""
This script performs biclustering analysis on genomic data.
It processes input files and generates results based on specified parameters.
"""

import argparse
import time
import os
import numpy as np
import pandas as pd
import csv
import sys
from numba import jit, njit
from scipy.stats import hypergeom

# Define global variable
log_path = ""
log_path_chunk = ""
result_path = ""
result_trimmed_path = ""

def log(log_path, message):
    """Logs a message to the specified log file."""
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")

def parse_arguments():
    """
    Parses and validates command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Perform biclustering analysis on genomic data."
    )

    parser.add_argument(
        "-c1", "--chrom1",
        type=str,
        required=True,
        help="Required: Chromosome 1 identifier."
    )

    parser.add_argument(
        "-c2", "--chrom2",
        type=str,
        required=True,
        help="Required: Chromosome 2 identifier."
    )

    parser.add_argument(
        "-b", "--bim_file",
        type=str,
        default="ABCD_5099_qced.bim",
        help="bim file containing SNP locations for each chromosome."
    )

    parser.add_argument(
        "-i", "--interaction_file",
        type=str,
        help="Path to the interaction file. Default will be set based on chrom1 and chrom2."
    )

    parser.add_argument(
        "-o", "--out_dir",
        type=str,
        default="out/",
        help="Output directory to save results."
    )

    parser.add_argument(
        "-m", "--max_inter_length",
        type=int,
        default=60,
        help="Maximum interaction length. Default is 60."
    )

    parser.add_argument(
        "-p", "--pval_cutoff",
        type=float,
        default=1e-10,
        help="P-value cutoff for significance. Default is 1e-10."
    )

    parser.add_argument(
        "-t", "--test",
        type=int,
        help="Test number, for testing purposes only."
    )

    args = parser.parse_args()

    # Set default value for interaction_file based on chrom1 and chrom2
    if args.interaction_file is None:
        args.interaction_file = f"../../ehlers/jiliu/epistasis_results/epistasis_results_chr{args.chrom1}_chr{args.chrom2}.epi.qt"

    # Validate chromosome identifiers
    if not args.chrom1 or not args.chrom2:
        parser.error("Both chromosome identifiers must be provided.")

    # Set the global log_path variable based on the parsed arguments
    global log_path, log_path_chunk, result_path, result_trimmed_path

    log_path = os.path.join(args.out_dir, f"chr{args.chrom1}_chr{args.chrom2}_log.txt")
    log_path_chunk = os.path.join(args.out_dir, f"chr{args.chrom1}_chr{args.chrom2}_chunk.csv")

    if args.test:
        result_path = os.path.join(args.out_dir, f"test{args.test}_pval_results.csv")
        result_trimmed_path = os.path.join(args.out_dir, f"test{args.test}_pval_results_trimmed.csv")
    else:
        if args.chrom1 == args.chrom2:
            result_path = os.path.join(args.out_dir, f'chr{args.chrom1}_pval_results.csv')
            result_trimmed_path = os.path.join(args.out_dir, f'chr{args.chrom1}_pval_results_trimmed.csv')
        else:
            result_path = os.path.join(args.out_dir, f'chr{args.chrom1}_chr{args.chrom2}_pval_results.csv')
            result_trimmed_path = os.path.join(args.out_dir, f'chr{args.chrom1}_chr{args.chrom2}_pval_results_trimmed.csv')

    return args


def parse_plink_bim_file(bim_file: str, chrom1: str, chrom2: str):
    chrom1_snps = []
    chrom2_snps = []
    
    with open(bim_file, "r") as csv_file:
        for line in csv_file:
            row = line.split(",")
        
            if chrom1 == row[0]:
                chrom1_snps.append(row[1])
            if chrom2 == row[0]:
                chrom2_snps.append(row[1])

    return chrom1_snps, chrom2_snps


def parse_remma_interaction_data(interaction_file: str, pval_cutoff: float, chrom1: str, chrom2: str):
    interactions = dict()
    
    with open(interaction_file) as csv_file:
        header = True
        for line in csv_file:
            if header:
                print(line)
                header = False
                continue

            fields = line.split(",")
            chrom_1 = fields[5]  # CHR1
            snp_1 = fields[0]    # SNP1
            chrom_2 = fields[9]  # CHR2
            snp_2 = fields[1]    # SNP2
            p_value = float(fields[-2])  # P-value

            if (p_value < pval_cutoff and
                ((chrom_1 == chrom1 and chrom_2 == chrom2) or 
                 (chrom_2 == chrom1 and chrom_1 == chrom2))):
                    
                if (snp_1, snp_2) in interactions:
                    if p_value < interactions[(snp_1, snp_2)]:
                        interactions[(snp_1, snp_2)] = p_value
                elif (snp_2, snp_1) in interactions:
                    if p_value < interactions[(snp_2, snp_1)]:
                        interactions[(snp_2, snp_1)] = p_value
                else:
                    interactions[(snp_1, snp_2)] = p_value
    print(interactions)
    return interactions


def initialize_matrices2(bim_file: str, interaction_files: str, chrom1: str, chrom2: str, 
                         pval_cutoff: float):

    print("parsing files...")
    chrom1_snps, chrom2_snps = parse_plink_bim_file(bim_file, chrom1, chrom2)


    interactions = parse_remma_interaction_data(interaction_files, pval_cutoff, chrom1, chrom2)
    upper_triangular = chrom1 == chrom2


    N = 0
    n = 0

    total_markers1 = len(chrom1_snps)
    total_markers2 = len(chrom2_snps)

    if upper_triangular:
        N = sum(range(total_markers1))  # Sum of first total_markers1-1 integers
    else:
        N = total_markers1 * total_markers2

    significant_interactions = np.zeros((30000, 2), dtype=np.int64) # TODO: need to monitor the array size

    idx = 0
    
    snp1_dict = {snp.strip():i for i, snp in enumerate(chrom1_snps)}
    snp2_dict = {snp.strip():j for j, snp in enumerate(chrom2_snps)}
    for row in interactions.keys():
        significant_interactions[idx, 0] = snp1_dict[row[0]]
        significant_interactions[idx, 1] = snp2_dict[row[1]]
        idx += 1
        n += 1

    significant_interactions = significant_interactions[:idx]

    print("Interaction data initialized.")
    print("N: ", N)
    print("n: ", n)
    print("Chromosome1: ", chrom1)
    print("Chromosome2: ", chrom2)
    print("P-Value Cutoff: ", pval_cutoff)

    log(log_path, f"Interaction data initialized \nN: {N} \nn: {n} \nChr1: {chrom1} \nChr2: {chrom2} \nP-val Cutoff: {pval_cutoff} \n")

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
                         upper_triangular: bool, N: int, n: int, max_inter_length: int, min_interactions = 2):
    
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
                            
                            if k < min_interactions: 
                                k_is_zero = True
                                a_thresh = a
                                b_thresh = b
                        
                        if k > (m * n / N) and k >= min_interactions:
                            pval_results[(i, j, a, b)] = (k, m)
                        


    return pval_results


def process_significant_interactions(significant_interactions, max_inter_length, N, n, upper_triangular):
    """
    Processes significant interactions by creating submatrices around each (i, j),
    then merging any overlapping submatrices into a single larger region.
    Finally, calls compute_k_m_parallel on each merged region to compute hypergeometric
    statistics for sub-intervals within each merged region.

    :param significant_interactions: np.ndarray of shape (num_sig, 2) with (i, j) indices
    :param max_inter_length: int, maximum extension in each dimension for building submatrices
    :param N: int, total number of pairs (depending on same-chromosome or not)
    :param n: int, total number of significant interactions
    :param upper_triangular: bool, indicates if chrom1 == chrom2 (upper triangular indexing)
    :return: (results, merged_submatrices)
             where results is a dict of ((i, j, a, b) -> (k, m))
             and merged_submatrices is a list of merged 2D intervals
    """

    # 1) Build initial list of submatrices from each (i, j).
    initial_submatrices = []
    for (i, j) in significant_interactions:
        i_start = max(i - max_inter_length + 1, 0)
        i_end   = i + max_inter_length
        j_start = max(j - max_inter_length + 1, 0)
        j_end   = j + max_inter_length
        initial_submatrices.append((i_start, i_end, j_start, j_end))

    # 2) Merge overlapping submatrices in a multi-pass manner
    def overlap_2d(subA, subB):
        """
        Checks if submatrix A overlaps submatrix B in 2D.
        subA, subB each is (i_start, i_end, j_start, j_end).
        """
        (Ai_start, Ai_end, Aj_start, Aj_end) = subA
        (Bi_start, Bi_end, Bj_start, Bj_end) = subB

        # No overlap if one is completely to the left or right or above/below the other
        if Ai_end   < Bi_start or Bi_end   < Ai_start: return False
        if Aj_end   < Bj_start or Bj_end   < Aj_start: return False
        return True

    def merge_2d(subA, subB):
        """
        Merges two overlapping submatrices A and B into one bounding submatrix.
        """
        (Ai_start, Ai_end, Aj_start, Aj_end) = subA
        (Bi_start, Bi_end, Bj_start, Bj_end) = subB
        return (min(Ai_start, Bi_start),
                max(Ai_end,   Bi_end),
                min(Aj_start, Bj_start),
                max(Aj_end,   Bj_end))

    changed = True
    merged_submatrices = initial_submatrices[:]

    merge_cnt = 0  # For optional reporting

    while changed:
        changed = False
        new_list = []

        for current_submatrix in merged_submatrices:
            if not new_list:
                # If new_list is empty, just add
                new_list.append(current_submatrix)
            else:
                # Try to merge with any existing submatrix in new_list
                merged_any = False
                for idx in range(len(new_list)):
                    if overlap_2d(current_submatrix, new_list[idx]):
                        # Merge them
                        new_list[idx] = merge_2d(current_submatrix, new_list[idx])
                        merge_cnt += 1
                        merged_any = True
                        changed = True
                        break

                if not merged_any:
                    new_list.append(current_submatrix)

        merged_submatrices = new_list

    print("Merge Count:", merge_cnt)
    log(log_path, f"Merge Count: {merge_cnt}")

    # 3) For each fully merged submatrix, call compute_k_m_parallel
    results = {}
    for (mi_start, mi_end, mj_start, mj_end) in merged_submatrices:
        chunk_result = compute_k_m_parallel(
            mi_start, mi_end, 
            mj_start, mj_end,
            significant_interactions, 
            upper_triangular, 
            N, n, 
            max_inter_length
        )
        print(f"i_start: {mi_start}; i_end: {mi_end}; "
              f"j_start: {mj_start}; j_end: {mj_end}. "
              f"Length: {len(chunk_result)}")
        log(log_path_chunk, f"{mi_start},{mi_end},{mj_start},{mj_end},{len(chunk_result)}")
        # Merge chunk_result into a single dictionary
        results.update(chunk_result)

    return results, merged_submatrices



def compute_interval_pval(pval_results: dict, N: int, n: int, out_dir: str, chrom1: str, chrom2: str):
    # correction = 0.05 / ((N ** 2) * 0.5)
    correction = 0.05

    if len(pval_results) > 0:
        correction = 0.05 / len(pval_results)
    
    log(log_path, f"Correction: {round(correction, 4)}")

    computed_pvals = {}
    
    with open(result_path, 'w') as writer:
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
    
    print(f"Computed p-values and saved to file as {result_path}.")
    log(log_path, f"Computed p-values and saved to file as {result_path}.")
    return pval_results

@njit
def check_overlap(i1, j1, a1, b1, i2, j2, a2, b2):
    if i1 == i2 and j1 == j2 and a1 == a2 and b1 == b2:
        return True

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

def process_pval_file(chrom1, chrom2, out_dir):


    print(f"Reading p-value results from {result_path}...", flush=True)
    log(log_path, f"Reading p-value results from {result_trimmed_path}...")

    # Read the CSV file into NumPy arrays
    data = np.genfromtxt(result_path, delimiter=',', skip_header=1)
    # Adjust indices if needed based on your CSV structure
    i = data[:, 0].astype(np.int32)
    j = data[:, 1].astype(np.int32)
    a = data[:, 2].astype(np.int32)
    b = data[:, 3].astype(np.int32)
    p_values = data[:, 6]

    n = len(i)
    print(f"Total intervals read: {n}", flush=True)
    log(log_path, f"Total intervals read: {n}")
    # Stack intervals into a single array
    intervals = np.stack((i, j, a, b), axis=1)

    # Sort intervals by p-value (ascending order)
    sorted_indices = np.argsort(p_values)
    intervals = intervals[sorted_indices]
    p_values = p_values[sorted_indices]

    print("Trimming overlapping intervals...", flush=True)
    log(log_path, "Trimming overlapping intervals...")

    # Call the njit-compiled function to trim intervals
    keep = trim_intervals(intervals, p_values)

    # Select intervals to keep
    trimmed_intervals = intervals[keep]
    trimmed_p_values = p_values[keep]

    print(f"Total intervals after trimming: {len(trimmed_intervals)}", flush=True)
    log(log_path, f"Total intervals after trimming: {len(trimmed_intervals)}")

    # Write the trimmed intervals to the output file
    print(f"Writing trimmed intervals to {result_trimmed_path}...", flush=True)
    log(log_path, f"Writing trimmed intervals to {result_trimmed_path}...")
    with open(result_trimmed_path, 'w') as f_out:
        f_out.write('i,j,a,b,p_value\n')
        for interval, p_value in zip(trimmed_intervals, trimmed_p_values):
            i_out, j_out, a_out, b_out = interval
            f_out.write(f"{i_out},{j_out},{a_out},{b_out},{p_value}\n")

    print(f"File trim completed, trimmed file saved as {result_trimmed_path}", flush=True)
    log(log_path, f"File trim completed, trimmed file saved as {result_trimmed_path}")

def main():
    """
    Main execution function that performs biclustering analysis.
    
    Workflow:
    1. Parses command-line arguments
    2. Initializes matrices and processes data
    3. Outputs results to output.csv
    """
    args = parse_arguments()

    # Configuration settings from arguments
    chrom1 = args.chrom1
    chrom2 = args.chrom2
    bim_file = args.bim_file
    interaction_file = args.interaction_file
    out_dir = args.out_dir
    max_inter_length = args.max_inter_length
    pval_cutoff = args.pval_cutoff
    test_num =args.test
    
    # Clear existing logs
    with open(log_path, "w") as file:
        file.write("")
    with open(log_path_chunk, "w") as file:
        file.write("i_start,i_end,j_start,j_end,chunk_size\n")

    start_time = time.time()


    print(bim_file, interaction_file, out_dir)

    print(f"Start Biclustering Analysis.\nChrom 1: {chrom1}\nChrom 2: {chrom2}")
    log(log_path, f"Start Biclustering Analysis.\nChrom 1: {chrom1}\nChrom 2: {chrom2}")
    # Initialize matrices
    N, n, significant_interactions, upper_triangular, chrom1_snps, chrom2_snps = initialize_matrices2(
        bim_file, interaction_file, chrom1, chrom2, pval_cutoff
    )

    init_complete_time = time.time()
    elapsed_time = init_complete_time - start_time
    print(f"Matrices initialized: {elapsed_time // 60} minutes, {int(elapsed_time) % 60} seconds")
    log(log_path, f"Matrices initialized: {elapsed_time // 60} minutes, {int(elapsed_time) % 60} seconds")

    # Process significant interactions
    if n == 0:
        pval_results = {}
    else:
        pval_results, _ = process_significant_interactions(significant_interactions, max_inter_length, N, n, upper_triangular)

    interactions_complete_time = time.time()
    elapsed_time = interactions_complete_time - init_complete_time
    print(f"Interactions processed: {elapsed_time // 60} minutes, {int(elapsed_time) % 60} seconds")
    log(log_path, f"Interactions processed: {elapsed_time // 60} minutes, {int(elapsed_time) % 60} seconds")

    # Write results
    compute_interval_pval(pval_results, N, n, out_dir, chrom1, chrom2)

    result_write_time = time.time()
    elapsed_time = result_write_time - interactions_complete_time
    print(f"Results written: {elapsed_time // 60} minutes, {int(elapsed_time) % 60} seconds")
    log(log_path, f"Results written: {elapsed_time // 60} minutes, {int(elapsed_time) % 60} seconds")

    # Trim Intervals
    try:
        process_pval_file(chrom1, chrom2, out_dir)
    except:
        print("No interaction found")

    trim_complete_time = time.time()
    elapsed_time = trim_complete_time - result_write_time
    print(f"Results trimmed: {elapsed_time // 60} minutes, {int(elapsed_time) % 60} seconds")
    log(log_path, f"Results trimmed: {elapsed_time // 60} minutes, {int(elapsed_time) % 60} seconds")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds") 
    log(log_path, f"Elapsed time: {elapsed_time} seconds")


if __name__ == '__main__':
    main()
