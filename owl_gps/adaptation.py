"""Single-stage direct batch optimization used for OWL-GPS online adaptation."""
from dataclasses import dataclass
import torch
from .losses import combined_loss

@dataclass
class UpdateResult:
    loss: float
    samples: int
    optimizer_steps: int

class OnlineAdapter:
    """Direct Adam updates over relevance-selected acquired observations.

    No bilevel train/validation split and no separate newest-sample step.
    The original research code accumulates gradients over microbatches and
    runs 1 step normally or 5 steps if core>=30 and counter%10==0.
    """

    def __init__(self, model, *, lr=1e-3, microbatch=1, kl_weight=1e-4,
                 device="cpu"):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.microbatch = microbatch
        self.kl_weight = kl_weight
        self.optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)

    def update(self, selected, *, steps=1, noise_weight_fn=None):
        if not selected:
            return UpdateResult(0., 0, 0)
        if self.microbatch < 1:
            raise ValueError("microbatch must be positive")
        self.model.train()
        total_loss = 0.
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            chunks = [selected[i:i+self.microbatch] for i in range(0,len(selected),self.microbatch)]
            step_loss = 0.
            for chunk in chunks:
                inputs = torch.stack([x.image for x in chunk]).to(self.device)
                labels = torch.stack([x.pseudo_label for x in chunk]).to(self.device)
                a,b,mu,sigma = self.model(inputs)
                weights = (None if noise_weight_fn is None
                           else torch.stack([noise_weight_fn(x) for x in chunk]).to(self.device))
                loss = combined_loss(a,b,labels,mu,sigma,
                                     noise_weights=weights,kl_weight=self.kl_weight)
                (loss/len(chunks)).backward()
                step_loss += float(loss.detach())/len(chunks)
            self.optimizer.step()
            total_loss += step_loss
        return UpdateResult(total_loss/steps, len(selected), steps)
