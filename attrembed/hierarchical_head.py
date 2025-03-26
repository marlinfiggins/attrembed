import torch
import torch.nn as nn


class HierarchicalClassifierHead(nn.Module):
    def __init__(self, input_dim, hierarchy_config, hidden_dim=128):
        super().__init__()
        self.classifiers = nn.ModuleList()
        for level in hierarchy_config:
            self.classifiers.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, level["num_classes"]),
                )
            )

    def forward(self, x):
        return [clf(x) for clf in self.classifiers]

    def get_embedding(self, x):
        with torch.no_grad():
            logits = [clf(x) for clf in self.classifiers]
            return torch.cat(logits, dim=-1)
