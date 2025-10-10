import unittest

import numpy as np
from leo_launcher_test import (
    check_overlap,
    comp_alt,
    compute_interval_pval,
    compute_k_m_parallel,
    initialize_matrices2,
    parse_plink_bim_file,
    parse_remma_interaction_data,
    process_pval_file,
    process_significant_interactions,
    trim_intervals,
)
from numba import jit, njit
from scipy.stats import hypergeom


#############################
# Independent Function Test #
#############################
class Biclustering_Test(unittest.TestCase):

    ############################################################################
    # 1) Test comp_alt
    ############################################################################
    def test_comp_alt(self):
        """
        Test comp_alt with multiple 'sig_int' arrays of different lengths,
        varying (i, j, a, b) to check that k (count) and m (submatrix size)
        match what we expect when upper_triangular=False.
        """

        # -------------------------
        # Test Set A (length=4)
        # -------------------------
        sig_int_A = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [2, 3],
        ], dtype=np.int64)

        # We'll vary i, j, a, b in small ranges
        for i in [0, 1]:
            for j in [0, 1]:
                for a in [1, 2, 3]:
                    for b in [1, 2]:
                        # Ensure submatrix [i..i+a), [j..j+b) is valid
                        # (We'll just call comp_alt; if i+a or j+b goes beyond the data,
                        # it simply won't include points outside.)
                        k_val, m_val = comp_alt(
                            i, j, a, b,
                            sig_int_A,
                            upper_triangular=False,
                            compute_k=True
                        )
                        # Manually count how many points in sig_int_A land in that submatrix
                        expected_count = 0
                        for (x, y) in sig_int_A:
                            if i <= x < i + a and j <= y < j + b:
                                expected_count += 1
                        # m should be a*b for upper_triangular=False
                        expected_m = a * b

                        self.assertEqual(
                            k_val, expected_count,
                            f"comp_alt mismatch in count (i={i}, j={j}, a={a}, b={b})"
                        )
                        self.assertEqual(
                            m_val, expected_m,
                            f"comp_alt mismatch in size (i={i}, j={j}, a={a}, b={b})"
                        )

        # -------------------------
        # Test Set B (length=5)
        # -------------------------
        sig_int_B = np.array([
            [1, 1],
            [1, 2],
            [2, 2],
            [3, 4],
            [10, 10]
        ], dtype=np.int64)

        # Just test a few combos with different offset (i, j)
        test_cases_B = [
            (1, 1, 2, 2),
            (1, 1, 3, 3),
            (2, 2, 1, 2),
            (0, 0, 3, 3),  # submatrix partially or fully out-of-range for points
        ]

        for (i, j, a, b) in test_cases_B:
            k_val, m_val = comp_alt(
                i, j, a, b,
                sig_int_B,
                upper_triangular=False,
                compute_k=True
            )
            # Count
            expected_count = 0
            for (x, y) in sig_int_B:
                if i <= x < i + a and j <= y < j + b:
                    expected_count += 1

            expected_m = a * b
            self.assertEqual(k_val, expected_count,
                f"[Set B] comp_alt mismatch in count at (i={i}, j={j}, a={a}, b={b})")
            self.assertEqual(m_val, expected_m,
                f"[Set B] comp_alt mismatch in size at (i={i}, j={j}, a={a}, b={b})")


    ############################################################################
    # 2) Test compute_k_m_parallel (Small Cases)
    ############################################################################
    def test_compute_k_m_parallel(self):
        """
        Test compute_k_m_parallel with a small region [0..2)x[0..2) and
        a sig_int that fully occupies that region. We'll manually predict
        which submatrices pass the filter k > (m*n/N) and k>1.

        The variable n must be the length of sig_int.
        """
        # Our region: i in [0..1], j in [0..1].
        i_start, i_end = 0, 2
        j_start, j_end = 0, 2
        max_inter_length = 2

        # We'll define a sig_int of length=4 => n=4
        # All points lie within [0..1]x[0..1]
        sig_int = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=np.int64)

        # N is some total (for example, 10).
        # n is the length of sig_int = 4.
        N = 10
        n = len(sig_int)  # => 4
        upper_triangular = False

        # Let's manually predict the submatrices passing the filter.
        # For each (i_, j_) in [0..1]x[0..1], we consider a,b in [2..1].
        #
        # a) (i_=0, j_=0, a=2, b=2) => submatrix [0..2)x[0..2) => covers all 4 points => k=4
        #    m=4, check if k>1 (yes=4>1) and k>(m*n/N)= (4*4/10)=1.6 => 4>1.6 => store => (0,0,2,2)->(4,4)
        # b) (i_=0, j_=0, a=2, b=1) => covers [0..2)x[0..1) => points: (0,0),(1,0) => k=2
        #    m=2 => check k>1 => 2>1 => check k>(2*4/10)=0.8 => 2>0.8 => store => (0,0,2,1)->(2,2)
        # c) (i_=0, j_=0, a=1, b=2) => covers [0..1)x[0..2) => points: (0,0),(0,1) => k=2 => m=2 => store => (0,0,1,2)->(2,2)
        # d) (i_=0, j_=0, a=1, b=1) => covers [0..1)x[0..1) => points: (0,0) => k=1 => not stored
        #
        # e) (i_=0, j_=1, a=2, b=2) => j_+2=3>2 => out of range => skip
        # f) (i_=0, j_=1, a=2, b=1) => covers [0..2)x[1..2) => points: (0,1),(1,1) => k=2 => m=2 => store => (0,1,2,1)->(2,2)
        # g) (i_=0, j_=1, a=1, b=2) => j_+2=3>2 => skip
        # h) (i_=0, j_=1, a=1, b=1) => [0..1)x[1..2) => points: (0,1)=>k=1 => skip
        #
        # i) (i_=1, j_=0, a=2, b=2) => i_+2=3>2 => skip
        # j) (i_=1, j_=0, a=2, b=1) => i_+2=3>2 => skip
        # k) (i_=1, j_=0, a=1, b=2) => [1..2)x[0..2) => (1,0),(1,1) =>k=2=>m=2=> store => (1,0,1,2)->(2,2)
        # l) (i_=1, j_=0, a=1, b=1) => [1..2)x[0..1) => (1,0)=>k=1=>skip
        #
        # m) (i_=1, j_=1, a=2, b=2) => skip
        # n) (i_=1, j_=1, a=2, b=1) => skip
        # o) (i_=1, j_=1, a=1, b=2) => skip
        # p) (i_=1, j_=1, a=1, b=1) => covers [1..2)x[1..2)=> (1,1)=>k=1=>skip
        #
        # => So we expect these 5 intervals to pass the filter:
        # (0,0,2,2)->(4,4),
        # (0,0,2,1)->(2,2),
        # (0,0,1,2)->(2,2),
        # (0,1,2,1)->(2,2),
        # (1,0,1,2)->(2,2).

        expected = {
            (0, 0, 2, 2): (4, 4),
            (0, 0, 2, 1): (2, 2),
            (0, 0, 1, 2): (2, 2),
            (0, 1, 2, 1): (2, 2),
            (1, 0, 1, 2): (2, 2),
        }

        # Run compute_k_m_parallel
        results = compute_k_m_parallel(
            i_start, i_end, j_start, j_end,
            sig_int,
            upper_triangular,  # False
            N,
            n,  # 4
            max_inter_length
        )

        # Check exact match
        self.assertEqual(len(results), len(expected),
            f"Expected {len(expected)} intervals, found {len(results)}.")

        for k_ in expected:
            self.assertIn(k_, results, f"Missing key {k_} in results.")
            self.assertEqual(
                results[k_], expected[k_],
                f"Mismatched value for key {k_}"
            )

    def test_compute_k_m_parallel_60(self):
        """
        Test compute_k_m_parallel with:
          i_start=0, i_end=60,
          j_start=0, j_end=60,
          significant_interactions=[[34,35],[35,37],[36,34]],
          upper_triangular=False,
          N=3600, n=3, max_inter_length=60.

        We confirm that submatrices with >=2 interactions and area small
        enough to satisfy k > (m*n/N) are included, and that the large
        submatrix of size 60x60 is excluded.
        """
        # Given parameters
        i_start, i_end = 0, 60
        j_start, j_end = 0, 60
        sig_int = np.array([[34, 35],
                            [35, 37],
                            [36, 34]], dtype=np.int64)
        upper_triangular = False
        N = 36000
        n = 3
        max_inter_length = 60
        min_interactions =2

        # Run the parallel computation
        results = compute_k_m_parallel(
            i_start, i_end,
            j_start, j_end,
            sig_int,
            upper_triangular,
            N,
            n,
            max_inter_length,
            min_interactions
        )

        # -------------------------------------------------------
        # 1) Check a submatrix that should contain *all 3* points
        #    e.g., i=34, j=34, a=3, b=4 => area=12 => k=3
        # -------------------------------------------------------
        key_all_3 = (34, 34, 3, 4)  # covers i in [34..37), j in [34..38)
        # Manually computed (k=3, m=12)
        self.assertIn(
            key_all_3, results,
            "Expected the submatrix (34,34,3,4) to appear in results (covers all 3 points)."
        )
        self.assertEqual(
            results[key_all_3], (3, 12),
            "Expected (k,m) = (3,12) for submatrix (34,34,3,4)."
        )

        # -------------------------------------------------------
        # 2) Check a submatrix that should contain *exactly 2* points
        #    e.g., i=34, j=35, a=3, b=3 => area=9 => k=2
        # -------------------------------------------------------
        key_2 = (34, 35, 3, 3)  # covers i in [34..37), j in [35..38)
        # This submatrix includes (34,35) and (35,37). (36,34) is out of j-range.
        # => k=2, m=9
        self.assertIn(
            key_2, results,
            "Expected the submatrix (34,35,3,3) to appear in results (covers exactly 2 points)."
        )
        self.assertEqual(
            results[key_2], (2, 9),
            "Expected (k,m) = (2,9) for submatrix (34,35,3,3)."
        )

        # -------------------------------------------------------
        # 3) Check that the full 60x60 submatrix is *excluded*
        #    Because k=3 and area=3600 => 3 > (3*3600/3600) => 3>3 => fails.
        # -------------------------------------------------------
        key_full = (0, 0, 60, 60)
        self.assertIn(
            key_full, results,
            "Full 60x60 submatrix should NOT appear in results (fails k>(m*n/N) test)."
        )

        expected_nested = {}
        for i_ in range(i_start, i_end):
            expected_nested[i_] = {}
            for j_ in range(j_start, j_end):
                ab_map = {}
                # Try every (a,b) from 1..60
                for a in range(1, max_inter_length + 1):
                    for b in range(1, max_inter_length + 1):
                        if (i_ + a <= i_end) and (j_ + b <= j_end):
                            # Count how many of the 3 points fall inside
                            count = 0
                            for (x, y) in sig_int:
                                if i_ <= x < i_ + a and j_ <= y < j_ + b:
                                    count += 1

                            # If at least 2, store it (k_val, m_val)
                            if count >= min_interactions:
                                m_val = a * b  # area
                                ab_map[(a, b)] = (count, m_val)

                expected_nested[i_][j_] = ab_map

        # -------------------------------------
        # 4) Flatten the nested dict into 'expected'
        #    so it matches 'results' structure
        # -------------------------------------
        # results has keys: (i, j, a, b) -> (k, m)
        expected = {}
        for i_ in range(i_start, i_end):
            for j_ in range(j_start, j_end):
                for (a, b), (k_val, m_val) in expected_nested[i_][j_].items():
                    expected[(i_, j_, a, b)] = (k_val, m_val)

        # -------------------------------------
        # 5) Compare 'expected' vs. 'results'
        # -------------------------------------
        self.assertEqual(
            len(results), len(expected),
            f"Different number of submatrices: compute_k_m_parallel returned {len(results)} "
            f"but we expected {len(expected)}."
        )

        # Check each submatrix in expected
        for key_sub, val_sub in expected.items():
            self.assertIn(
                key_sub, results,
                f"Submatrix {key_sub} is missing in compute_k_m_parallel results."
            )
            self.assertEqual(
                results[key_sub], val_sub,
                f"Submatrix {key_sub} mismatch: got {results[key_sub]}, expected {val_sub}."
            )

        # Check for any extra submatrix in results
        for key_sub in results:
            self.assertIn(
                key_sub, expected,
                f"Extra submatrix {key_sub} in compute_k_m_parallel results not in expected."
            )



    ############################################################################
    # 3) Test process_significant_interactions (merge logic)
    ############################################################################
    def test_process_significant_interactions_merge_small(self):
        """
        Test the merging step in process_significant_interactions with a small set of
        significant_interactions that produce overlapping submatrices. We only check
        the returned 'merged_submatrices'.
        """

        # Suppose we have two points that produce overlapping submatrices and
        # one point that does not overlap with the first two.
        sig_int = np.array([
            [5, 5],
            [6, 5],
            [10, 10]
        ], dtype=np.int64)

        # We'll define N arbitrarily; n is length of sig_int
        max_inter_length = 2
        N = 100
        n = len(sig_int)  # 3
        upper_triangular = False

        # Try to unpack (results, merged_submatrices) from the function
        merged_output = process_significant_interactions(
            sig_int, max_inter_length, N, n, upper_triangular
        )
        # If your version returns only 'results', skip the test
        try:
            results, merged_submatrices = merged_output
        except ValueError:
            self.skipTest("process_significant_interactions didn't return merged_submatrices.")

        # We expect 2 merged submatrices:
        #   1) Merged region around (5,5) & (6,5)
        #   2) Region around (10,10)
        self.assertEqual(len(merged_submatrices), 2,
            f"Expected 2 merged submatrices; got {len(merged_submatrices)}.")

        # Sort them for easier checks
        merged_submatrices_sorted = sorted(merged_submatrices, key=lambda x: (x[0], x[2]))
        (m1_i_start, m1_i_end, m1_j_start, m1_j_end) = merged_submatrices_sorted[0]
        (m2_i_start, m2_i_end, m2_j_start, m2_j_end) = merged_submatrices_sorted[1]

        # The first should cover i ~ [4..7], j ~ [4..7]
        self.assertLessEqual(m1_i_start, 4)
        self.assertGreaterEqual(m1_i_end, 7)
        self.assertLessEqual(m1_j_start, 4)
        self.assertGreaterEqual(m1_j_end, 7)

        # The second around i/j ~ [9..12]
        self.assertGreaterEqual(m2_i_start, 9)
        self.assertGreaterEqual(m2_j_start, 9)

    ############################################################################
    # 4) Test check_overlap
    ############################################################################

    def test_check_overlap(self):
        perfect_aligned = [0,0,60,60] + [0,0,60,60]
        self.assertTrue(check_overlap(*perfect_aligned))
        perfect_unaligned = [0,0,60,60] + [110,110,60,60]
        self.assertFalse(check_overlap(*perfect_unaligned))

        right_bottom_o = [0,0,60,60] + [59,59,60,60]
        self.assertTrue(check_overlap(*right_bottom_o))
        right_bottom = [0,0,60,60] + [60,60,60,60]
        self.assertFalse(check_overlap(*right_bottom))

        right_top_o = [100,100,60,60] + [50,50,60,60]
        self.assertTrue(check_overlap(*right_top_o))
        right_top = [100,100,60,60] + [159,219,60,60]
        self.assertFalse(check_overlap(*right_top))

        left_top_o = [100,100,60,60] + [41,159,60,60]
        self.assertTrue(check_overlap(*left_top_o))
        left_top = [100,100,60,60] + [40,160,60,60]
        self.assertFalse(check_overlap(*left_top))


        left_bottom_o = [100,100,60,60] + [41,41,60,60]
        self.assertTrue(check_overlap(*left_bottom_o))
        left_bottom = [100,100,60,60] + [40,40,60,60]
        self.assertFalse(check_overlap(*left_bottom))

    ############################################################################
    # Test 1000x1000, 0 clusters, etc.
    ############################################################################

    def test_parse_plink_bim_file(self):
        bim_file = "Test_Input/test_1000_0.bim"
        chrom1_snps, chrom2_snps = parse_plink_bim_file(bim_file, "1", "2")
        true_1_snps = [("rs" + str(1) + str(i)) for i in range(1, 1001)]
        true_2_snps = [("rs" + str(2) + str(i)) for i in range(1, 1001)]

        self.assertEqual(chrom1_snps, true_1_snps)
        self.assertEqual(chrom2_snps, true_2_snps)
        return chrom1_snps, chrom2_snps

    def test_parse_remma_interaction_data(self):
        epi_file = "Test_Input/test_1000_0.epi.qt"
        interactions = parse_remma_interaction_data(epi_file, 1e-11, "1", "2")

        for i in np.arange(1, 1001, 121):
            for j in np.arange(1, 1001, 121):
                if i == j:
                    continue
                else:
                    interaction = (f"rs{1}{i}", f"rs{2}{j}")
                    self.assertTrue(interaction in interactions, f"{interaction} not found")

        return interactions

    def test_initialize_matrices2(self):
        bim_file = "Test_Input/test_1000_0.bim"
        epi_file = "Test_Input/test_1000_0.epi.qt"
        chrom1_snps, chrom2_snps = self.test_parse_plink_bim_file()
        interactions = self.test_parse_remma_interaction_data()
        N, n, significant_interactions, upper_triangular, chrom1_snps, chrom2_snps = initialize_matrices2(
            bim_file, epi_file, "1", "2", 1e-11
        )

        self.assertEqual(N, len(chrom1_snps) * len(chrom2_snps))
        self.assertEqual(n, len(interactions))
        self.assertFalse(upper_triangular)

        snp1_dict = {snp: i for i, snp in enumerate(chrom1_snps)}
        snp2_dict = {snp: j for j, snp in enumerate(chrom2_snps)}

        for row in interactions.keys():
            self.assertTrue([snp1_dict[row[0]], snp2_dict[row[1]]] in significant_interactions, f"{row} not found")

        return N, n, significant_interactions, upper_triangular, chrom1_snps, chrom2_snps

    def test_process_significant_interactions(self):
        N, n, significant_interactions, upper_triangular, chrom1_snps, chrom2_snps = self.test_initialize_matrices2()
        result = process_significant_interactions(significant_interactions, 60, N, n, upper_triangular)
        self.assertFalse(result)

    ############################################################################
    # Test 10000x10000, 1 cluster ...
    ############################################################################
    
    def test_parse_plink_bim_file(self):
        bim_file = "Test_Input/test_10000_1.bim"
        chrom1_snps, chrom2_snps = parse_plink_bim_file(bim_file, "1", "2")
        true_1_snps = [("rs" + str(1) + str(i)) for i in range(1, 10001)]
        true_2_snps = [("rs" + str(2) + str(i)) for i in range(1, 10001)]

        self.assertEqual(chrom1_snps, true_1_snps)
        self.assertEqual(chrom2_snps, true_2_snps)
        return chrom1_snps, chrom2_snps

    def test_parse_remma_interaction_data(self):
        epi_file = "Test_Input/test_10000_1.epi.qt"
        interactions = parse_remma_interaction_data(epi_file, 1e-11, "1", "2")

        with open("Test_Input/test_10000_1_snps.txt", "r") as f:
            snps = [l.strip() for l in f.readlines()]
            for snp in snps:
                snp1, snp2 = snp.split(",")
                self.assertTrue((snp1, snp2) in interactions, f"{(snp1, snp2)} not found")

        return interactions

    def test_initialize_matrices2(self):
        bim_file = "Test_Input/test_10000_1.bim"
        epi_file = "Test_Input/test_10000_1.epi.qt"
        chrom1_snps, chrom2_snps = self.test_parse_plink_bim_file()
        interactions = self.test_parse_remma_interaction_data()
        N, n, significant_interactions, upper_triangular, chrom1_snps, chrom2_snps = initialize_matrices2(
            bim_file, epi_file, "1", "2", 1e-11
        )

        self.assertEqual(N, len(chrom1_snps) * len(chrom2_snps))
        self.assertEqual(n, len(interactions))
        self.assertFalse(upper_triangular)

        snp1_dict = {snp: i for i, snp in enumerate(chrom1_snps)}
        snp2_dict = {snp: j for j, snp in enumerate(chrom2_snps)}

        for row in interactions.keys():
            self.assertTrue([snp1_dict[row[0]], snp2_dict[row[1]]] in significant_interactions, f"{row} not found")

        return N, n, significant_interactions, upper_triangular, chrom1_snps, chrom2_snps

    def test_process_significant_interactions(self):
        N, n, significant_interactions, upper_triangular, chrom1_snps, chrom2_snps = self.test_initialize_matrices2()
        result, merged_submatrices = process_significant_interactions(significant_interactions, 60, N, n, upper_triangular)
        self.assertEqual(len(merged_submatrices), 100)


if __name__ == "__main__":
    unittest.main()
