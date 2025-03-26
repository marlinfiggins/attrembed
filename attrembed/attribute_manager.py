import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F

from classifier_head import ClassifierHead
from hierarchical_head import HierarchicalClassifierHead
from trainer import ClassifierTrainer, HierarchicalClassifierTrainer


class AttributeEmbeddingManager:
    def __init__(self, encoder, device="cpu"):
        self.encoder = encoder
        self.device = device

        self.classifiers = {}                     # attribute_name -> classifier
        self.embedding_store = defaultdict(dict)  # attribute -> {text_id: embedding}
        self.base_embeddings = {}                 # text_id -> base embedding
        self.id_to_text = {}                      # text_id -> original text

    def add_texts(self, ids, texts):
        embeddings = self.encoder.encode(texts).detach()
        for text_id, emb, text in zip(ids, embeddings, texts):
            self.base_embeddings[text_id] = emb.to(self.device)
            self.id_to_text[text_id] = text

    def load_from_dataframe(self, df: pd.DataFrame, embedding_col="vector", id_col="id", text_col="Text"):
        """
        Load base embeddings and metadata from a DataFrame.
        Expects embedding_col to be a list/array of floats per row.
        """
        assert embedding_col in df.columns
        assert id_col in df.columns
        assert text_col in df.columns

        for _, row in df.iterrows():
            text_id = row[id_col]
            emb = torch.tensor(row[embedding_col]).float().to(self.device)
            self.base_embeddings[text_id] = emb
            self.id_to_text[text_id] = row[text_col]

    def get_attribute_labels(self, df: pd.DataFrame, attribute_name: str, id_col="ID"):
        """
        Return: (ids, labels) for a given attribute column in the dataframe.
        """
        assert attribute_name in df.columns
        ids = df[id_col].tolist()
        labels = df[attribute_name].tolist()
        return ids, labels

    def train_classifier(self, attribute_name, ids, labels, num_classes,
                         hidden_dim=128, epochs=5, model_factory=None):
        embeddings = torch.stack([self.base_embeddings[i] for i in ids])
        input_dim = embeddings.size(1)

        custom_model = model_factory(input_dim, num_classes) if model_factory else None
        trainer = ClassifierTrainer(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            model=custom_model,
            device=self.device
        )
        classifier = trainer.train(embeddings, labels, epochs=epochs)

        self.classifiers[attribute_name] = classifier

    def train_hierarchical_classifier(self, attribute_name, ids, label_paths,
                                       hierarchy_config, hidden_dim=128, epochs=5,
                                       model_factory=None, use_masking=False):
        """
        Trains a hierarchical classifier head based on precomputed embeddings.

        attribute_name: name for this composite embedding space
        ids: list of sample IDs
        label_paths: list of Tensors [L][N] with label index per level
        hierarchy_config: metadata for each level (num_classes, mappings, etc.)
        """
        embeddings = torch.stack([self.base_embeddings[i] for i in ids])
        input_dim = embeddings.size(1)

        if model_factory:
            model = model_factory(input_dim, hierarchy_config)
        else:
            model = HierarchicalClassifierHead(
                input_dim=input_dim,
                hierarchy_config=hierarchy_config,
                hidden_dim=hidden_dim
            )

        trainer = HierarchicalClassifierTrainer(
            model=model,
            config=hierarchy_config,
            device=self.device
        )
        trained_model = trainer.train(
            embeddings,
            label_paths,
            epochs=epochs,
            use_masking=use_masking
        )
        self.classifiers[attribute_name] = trained_model

    def compute_attribute_embeddings(self, attribute_name, ids, batch_size=64):
        if attribute_name not in self.classifiers:
            raise ValueError(f"Classifier for attribute '{attribute_name}' not found.")

        classifier = self.classifiers[attribute_name]
        classifier.eval()

        with torch.no_grad():
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embs = torch.stack([self.base_embeddings[j] for j in batch_ids])
                logits = classifier(batch_embs)
                if isinstance(logits, torch.Tensor):
                    # Flat classifier
                    normed = F.normalize(logits, dim=1)
                elif isinstance(logits, list) and all(isinstance(l, torch.Tensor) for l in logits):
                    # Hierarchical: concat and normalize
                    logits_concat = torch.cat(logits, dim=1)
                    normed = F.normalize(logits_concat, dim=1)
                else:
                    raise ValueError(f"Unexpected logits format for '{attribute_name}': {type(logits)}")

                for j, emb in zip(batch_ids, normed):
                    self.embedding_store[attribute_name][j] = emb.cpu()

    def compute_all_attributes(self, ids, batch_size=64):
        result = {}
        for attr in self.classifiers.keys():
            self.compute_attribute_embeddings(attr, ids, batch_size=batch_size)
            result[attr] = {i: self.embedding_store[attr][i] for i in ids}
        return result

    def get_embedding(self, text_id, attribute_name):
        return self.embedding_store[attribute_name].get(text_id, None)

    def get_all_embeddings(self, attribute_name):
        return self.embedding_store[attribute_name]

    def find_top_k(self, query_id, attribute_name, k=5):
        query = self.get_embedding(query_id, attribute_name)
        if query is None:
            raise ValueError(f"No embedding for query ID '{query_id}' under attribute '{attribute_name}'.")

        all_items = self.embedding_store[attribute_name]
        ids = list(all_items.keys())
        embs = torch.stack([all_items[i] for i in ids])
        sims = F.cosine_similarity(query.unsqueeze(0), embs)
        topk = torch.topk(sims, k)

        return [(ids[i], sims[i].item(), self.id_to_text[ids[i]]) for i in topk.indices]

    def combined_similarity(self, query_id, attribute_names, weights=None, k=5):
        ids = list(self.base_embeddings.keys())
        sims_total = None

        for i, attr in enumerate(attribute_names):
            query = self.get_embedding(query_id, attr)
            if query is None:
                raise ValueError(f"No embedding for {query_id} under attribute {attr}")
            embs = torch.stack([self.embedding_store[attr][j] for j in ids])
            sims = F.cosine_similarity(query.unsqueeze(0), embs)
            w = weights[i] if weights else 1.0
            sims_total = sims * w if sims_total is None else sims_total + sims * w

        topk = torch.topk(sims_total, k)
        return [(ids[i], sims_total[i].item(), self.id_to_text[ids[i]]) for i in topk.indices]

    def save_classifier(self, attribute_name, path):
        model = self.classifiers[attribute_name]
        os.makedirs(path, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": model.model[0].in_features,
            "num_classes": model.model[-1].out_features
        }, os.path.join(path, f"{attribute_name}_classifier.pt"))

    def load_classifier(self, attribute_name, path, model_factory=None):
        full_path = os.path.join(path, f"{attribute_name}_classifier.pt")
        data = torch.load(full_path)
        model = model_factory(data["input_dim"], data["num_classes"]) if model_factory \
                else ClassifierHead(data["input_dim"], data["num_classes"])
        model.load_state_dict(data["state_dict"])
        model.to(self.device)
        self.classifiers[attribute_name] = model.eval()
