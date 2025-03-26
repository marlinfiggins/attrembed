import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def fit_similarity_regression(
    manager, ids, attribute_names, n_pairs=1000, seed=42, plot=True
):
    random.seed(seed)
    torch.manual_seed(seed)

    # Sample random ID pairs
    pairs = random.sample(
        [(i, j) for i in ids for j in ids if i < j],
        min(n_pairs, len(ids) * (len(ids) - 1) // 2),
    )

    X = []
    y = []

    for i, j in pairs:
        # Base space cosine similarity
        x_i = manager.base_embeddings[i]
        x_j = manager.base_embeddings[j]
        base_sim = F.cosine_similarity(x_i.unsqueeze(0), x_j.unsqueeze(0)).item()

        # Attribute similarities
        attr_sims = []
        for attr in attribute_names:
            z_i = manager.get_embedding(i, attr)
            z_j = manager.get_embedding(j, attr)
            sim = F.cosine_similarity(z_i.unsqueeze(0), z_j.unsqueeze(0)).item()
            attr_sims.append(sim)

        X.append(attr_sims)
        y.append(base_sim)

    X = np.array(X)
    y = np.array(y)

    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)

    if plot:
        # --- Plot 1: Attribute weights
        plt.figure(figsize=(6, 4))
        plt.bar(attribute_names, reg.coef_)
        plt.title(
            f"Attribute Weights for Reconstructing Base Similarity\n$R^2$ = {r2:.4f}"
        )
        plt.ylabel("Regression Weight")
        plt.xticks(rotation=15)
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.show()

        # --- Plot 2: True vs Predicted
        plt.figure(figsize=(5.5, 5))
        plt.scatter(y, y_pred, alpha=0.5, s=20)

        min_sim = min(min(y), min(y_pred))
        max_sim = max(max(y), max(y_pred))
        plt.plot(
            [min_sim, max_sim],
            [min_sim, max_sim],
            "--",
            color="gray",
            label="Ideal: y = x",
        )

        plt.xlabel("True Cosine Similarity (Base)")
        plt.ylabel("Predicted Cosine Similarity (Weighted)")
        plt.title(f"Similarity Reconstruction ($R^2$ = {r2:.4f})")
        plt.xlim(min_sim, max_sim)
        plt.ylim(min_sim, max_sim)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "regression_model": reg,
        "attribute_names": attribute_names,
        "weights": dict(zip(attribute_names, reg.coef_)),
        "intercept": reg.intercept_,
        "r2_score": r2,
    }
