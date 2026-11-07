"""Regression fixtures checked against the active GIA implementation
in the uploaded icml2019test(1).py (not its unused exhaustive variant).
"""
import unittest
import numpy as np
from owl_gps.clustering import (
    axis_coverage_sets, greedy_intersection_representative,
    choose_epsilon_for_target_coverage, greedy_intersection_committee,
    greedy_intersection_clusters,
)

class TestOriginalGIAParity(unittest.TestCase):
    def setUp(self):
        self.points = np.array([
            [0.,0.],[.05,.02],[.14,.05],[1.,1.],[1.02,1.10],
            [2.,0.],[2.1,.03],[3.,3.],[3.1,3.1],[3.05,3.02],
        ])

    def test_greedy_intersections_not_farthest_point_sampling(self):
        groups = greedy_intersection_committee(self.points,k=4,epsilon=.06)
        self.assertEqual(groups,[{7,8,9},{0,1,2},{3,4},{5,6}])

    def test_original_adaptive_epsilon(self):
        mapping = {i:v for i,v in enumerate(self.points)}
        epsilon,fraction = choose_epsilon_for_target_coverage(mapping)
        self.assertAlmostEqual(epsilon,.055)
        self.assertAlmostEqual(fraction,.3)

    def test_coverage_sets_and_representative(self):
        mapping = {i:v for i,v in enumerate(self.points)}
        self.assertEqual(len(axis_coverage_sets(mapping,.06)),2)
        representative,selected=greedy_intersection_representative(mapping,.06)
        self.assertEqual(selected,{7,8,9})
        np.testing.assert_allclose(representative,self.points[[7,8,9]].mean(axis=0))

    def test_original_pca_7d_fixture(self):
        sample=np.random.default_rng(42).normal(size=(8,33))
        groups=greedy_intersection_clusters(sample,k=4)
        self.assertEqual(groups,[{7},{3,5},{0,1,2,4},{6}])

    def test_duplicate_relevance_vectors_keep_memory_ids(self):
        data=np.zeros((16,33))
        data[8:]=1.
        groups=greedy_intersection_clusters(data,k=4)
        self.assertEqual(set().union(*groups),set(range(16)))
        self.assertEqual(sum(map(len,groups)),16)

    def test_small_buffer_coverage(self):
        for n in range(1,8):
            with self.subTest(n=n):
                data=np.arange(n*33,dtype=float).reshape(n,33)
                groups=greedy_intersection_clusters(data,k=4)
                self.assertEqual(set().union(*groups),set(range(n)))

if __name__=='__main__':
    unittest.main()
