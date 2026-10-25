"""CPU smoke demonstration. Synthetic labels are NOT PFAS results.

Run from repository root: python examples/synthetic_demo.py
"""
import numpy as np
import torch
from torch import nn
from owl_gps import BoundedMemory, Observation, Candidate, choose_candidates
from owl_gps.adaptation import OnlineAdapter


class TinyDemoModel(nn.Module):
    """Minimal model satisfying the online adapter API, not paper architecture."""
    def __init__(self, concepts=3):
        super().__init__()
        self.conv = nn.Conv2d(concepts, 8, 3, padding=1)
        self.classifier_a = nn.Conv2d(8, 1, 1)
        self.classifier_b = nn.Conv2d(8, 1, 1)
        self.mu = nn.Parameter(torch.zeros(concepts))
        self.log_sigma = nn.Parameter(torch.full((concepts,), -2.))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, image):
        x = image.squeeze(2)
        h = torch.relu(self.conv(x))
        a,b = self.classifier_a(h), self.classifier_b(h)
        mu = self.mu.unsqueeze(0).expand(image.size(0), -1)
        sigma = self.log_sigma.exp().unsqueeze(0).expand_as(mu)
        return a,b,mu,sigma


def main():
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    memory = BoundedMemory(core_capacity=10, reservoir_capacity=10)
    model = TinyDemoModel()
    adapter = OnlineAdapter(model, lr=1e-2)
    # Only features and model probabilities are available before querying.
    pool = [torch.randn(3,1,8,8) for _ in range(8)]
    unseen = set(range(len(pool)))
    for step in range(4):
        options = []
        with torch.no_grad():
            for i in sorted(unseen):
                a,_,mu,_ = model(pool[i].unsqueeze(0))
                options.append(Candidate(i,float(a.sigmoid().mean()),
                                         np.asarray(mu[0].detach())))
        chosen = choose_candidates(options,memory.references(),step=step,
                                   budget=4,phase="train",batch_size=1)[0]
        # Oracle access happens ONLY after acquisition; synthesize binary mask.
        index = chosen.index
        pseudo = (pool[index][0,0] > 0).long()
        measured = torch.full_like(pseudo,2)
        measured[0,0] = pseudo[0,0]
        memory.add(Observation(str(index),chosen.relevance,pool[index],
                               pseudo,measured))
        unseen.remove(index)
        update = adapter.update(memory.meta_batch(clusters=3))
        print(f"step={step+1} queried={index} meta_batch={update.samples} loss={update.loss:.4f}")


if __name__ == "__main__":
    main()
