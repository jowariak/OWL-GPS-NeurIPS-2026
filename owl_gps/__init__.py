"""OWL-GPS: relevance-guided sequential geospatial learning."""
from .acquisition import Candidate, choose_candidates, score_candidate
from .memory import BoundedMemory, Observation
from .metrics import evaluate_measured
__all__ = ["Candidate", "choose_candidates", "score_candidate",
           "BoundedMemory", "Observation", "evaluate_measured"]
