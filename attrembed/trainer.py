import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier_head import ClassifierHead
from torch.utils.data import DataLoader


class ClassifierTrainer:
    def __init__(self, input_dim, num_classes, device="cpu", model=None, hidden_dim=128):
        self.device = device
        if model is None:
            self.model = ClassifierHead(input_dim, num_classes, hidden_dim)
        else:
            self.model = model.to(device)

    def train(self, embeddings, labels, epochs=5, lr=1e-3):
        labels = torch.tensor(labels).to(self.device)
        embeddings = embeddings.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            logits = self.model(embeddings)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.model.eval()

    def predict_logits(self, embeddings):
        self.model.eval()
        with torch.no_grad():
            return self.model(embeddings.to(self.device))


def hierarchical_loss(
    logits, labels, hierarchy_config, loss_weights, kl_weights, use_masking=False
):
    total = 0.0
    ce = nn.CrossEntropyLoss()

    for i, (logit, target, cfg) in enumerate(zip(logits, labels, hierarchy_config)):
        if use_masking and i > 0:
            prev = labels[i - 1]
            mask = torch.zeros_like(logit).bool()
            for b in range(logit.size(0)):
                valid = cfg["children"][prev[b].item()]
                mask[b, valid] = True
            logit = logit.masked_fill(~mask, -1e9)
        total += loss_weights[i] * ce(logit, target)

        if i > 0 and not use_masking:
            soft_prev = F.softmax(logits[i - 1], dim=1)
            soft_curr = F.softmax(logit, dim=1)
            M = cfg["mapping"].to(logit.device)
            induced = torch.matmul(soft_prev, M)
            kl = F.kl_div(soft_curr.log(), induced, reduction="batchmean")
            total += kl_weights[i - 1] * kl

    return total

class HierarchicalClassifierTrainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.hierarchy_config = config
        self.device = device

    def train(
        self,
        embeddings,
        label_paths,
        epochs=5,
        batch_size=64,
        lr=3e-4,
        loss_weights=None,
        kl_weights=None,
        use_masking=False,
    ):

        if loss_weights is None:
            loss_weights = [1.0] * len(self.hierarchy_config)
        if kl_weights is None:
            kl_weights = [0.5] * (len(self.hierarchy_config) - 1)

        dataset = list(zip(embeddings, *label_paths))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            for batch in loader:
                batch = [b.to(self.device) for b in batch]
                emb = batch[0]
                labels = batch[1:]
                logits = self.model(emb)
                loss = hierarchical_loss(
                    logits,
                    labels,
                    self.hierarchy_config,
                    loss_weights,
                    kl_weights,
                    use_masking,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model


