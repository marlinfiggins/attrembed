from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F


def plot_similarity_heatmap(
    matrix,
    labels,
    title="Average Cosine Similarity",
    xlabel="Label B",
    ylabel="Label A",
    use_clustermap=True,
    vmin=-1.0,
    vmax=1.0,
):
    if use_clustermap:
        sns.clustermap(
            matrix,
            xticklabels=labels,
            yticklabels=labels,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            vmin=vmin,
            vmax=vmax,
        )
        plt.suptitle(title)
    else:
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            matrix,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
    plt.show()


def compute_similarity_matrix_from_manager(
    manager, ids, labels, attribute=None, label_names=None
):
    label_to_ids = defaultdict(list)
    for idx, label in zip(ids, labels):
        label_to_ids[label.item()].append(idx)

    unique_labels = sorted(label_to_ids.keys())
    matrix = np.zeros((len(unique_labels), len(unique_labels)))

    def get_embedding_batch(id_list):
        if attribute is None:
            return torch.stack([manager.base_embeddings[i] for i in id_list])
        return torch.stack([manager.get_embedding(i, attribute) for i in id_list])

    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            emb_i = get_embedding_batch(label_to_ids[label_i])
            emb_j = get_embedding_batch(label_to_ids[label_j])
            sims = F.cosine_similarity(emb_i.unsqueeze(1), emb_j.unsqueeze(0), dim=-1)
            matrix[i, j] = sims.mean().item()

    label_display = [label_names[i] if label_names else str(i) for i in unique_labels]
    return matrix, label_display


def compare_similarity_spaces(
    manager,
    ids,
    labels,
    label_names,
    attribute_name,
    level_name="Level",
    use_clustermap=False,
):
    matrix_base, display_labels = compute_similarity_matrix_from_manager(
        manager, ids, labels, attribute=None, label_names=label_names
    )
    plot_similarity_heatmap(
        matrix_base,
        display_labels,
        title=f"{level_name} – Base Embeddings",
        use_clustermap=use_clustermap,
    )

    matrix_attr, _ = compute_similarity_matrix_from_manager(
        manager, ids, labels, attribute=attribute_name, label_names=label_names
    )
    plot_similarity_heatmap(
        matrix_attr,
        display_labels,
        title=f"{level_name} – Attribute Embeddings",
        use_clustermap=use_clustermap,
    )

    return matrix_base, matrix_attr
