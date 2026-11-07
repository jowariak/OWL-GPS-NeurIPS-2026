import unittest
import numpy as np
from owl_gps.acquisition import Candidate, choose_candidates
from owl_gps.memory import BoundedMemory, Observation
from owl_gps.metrics import evaluate_measured


class TestComponents(unittest.TestCase):
    def test_selection_uses_features_not_labels(self):
        pool = [Candidate(i,0.5,np.array([float(i)])) for i in range(4)]
        result = choose_candidates(pool,[],step=0,budget=3,phase="train",batch_size=2)
        self.assertEqual(len(result),2)

    def test_memory_latest_is_selected(self):
        memory = BoundedMemory(core_capacity=2, reservoir_capacity=2)
        for i in range(3):
            memory.add(Observation(str(i),np.array([i],dtype=float),
                       None,None,None))
        keys = [s.key for s in memory.meta_batch(clusters=1, include_latest=True)]
        self.assertIn("2",keys)
        self.assertLessEqual(len(memory.core),2)
        self.assertLessEqual(len(memory.reservoir),2)

    def test_gia_clusters_cover_every_core_sample(self):
        from owl_gps.clustering import greedy_intersection_clusters
        vectors = np.asarray([[0.,0.], [.05,.02], [2.,2.], [2.1,2.]])
        groups = greedy_intersection_clusters(vectors, k=2)
        self.assertEqual(set().union(*groups), set(range(4)))
        self.assertTrue(all(groups))

    def test_meta_batch_preserves_original_core_and_reservoir_policy(self):
        memory = BoundedMemory(core_capacity=4, reservoir_capacity=3)
        for i in range(7):
            memory.add(Observation(str(i), np.array([float(i), 1.]),
                                   None, np.array([1]), None))
        batch = memory.meta_batch(clusters=2, extra_class0=0)
        self.assertEqual(len({s.key for s in batch}), len(batch))
        self.assertTrue(any(s in memory.core for s in batch))
        self.assertTrue(all(s in memory.core + memory.reservoir for s in batch))

    def test_mean_euclidean_not_squared_distance(self):
        from owl_gps.acquisition import novelty_score
        d = novelty_score(np.array([1.]), [np.array([0.]),np.array([3.])])
        self.assertAlmostEqual(d, 1.5)

    def test_measured_metrics_ignore_unknown(self):
        result = evaluate_measured([0,1,0,1],[0,1,2,0])
        self.assertEqual(result.confusion,((1,1),(0,1)))


if __name__ == "__main__":
    unittest.main()
