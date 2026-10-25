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
        keys = [s.key for s in memory.meta_batch(clusters=1)]
        self.assertIn("2",keys)
        self.assertLessEqual(len(memory.core),2)
        self.assertLessEqual(len(memory.reservoir),2)

    def test_measured_metrics_ignore_unknown(self):
        result = evaluate_measured([0,1,0,1],[0,1,2,0])
        self.assertEqual(result.confusion,((1,1),(0,1)))


if __name__ == "__main__":
    unittest.main()
