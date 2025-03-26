import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


def visualize_attribute_embeddings(
    manager, ids, labels_per_level, label_names_per_level, attribute_name=None
):
    """
    Visualizes 2D PCA projection of base or attribute-specific embeddings.

    Parameters:
        manager: AttributeEmbeddingManager
        ids: list of IDs to visualize
        labels_per_level: list of torch.Tensor, one per level (hierarchical or flat)
        label_names_per_level: list of list[str], names per level
        attribute_name: name of attribute to use, or None for base embeddings
    """
    # --- Retrieve embeddings
    if attribute_name is None:
        emb = torch.stack([manager.base_embeddings[i] for i in ids])
        title_prefix = "Base Embedding"
    else:
        emb = torch.stack([manager.get_embedding(i, attribute_name) for i in ids])
        title_prefix = f"Attribute: {attribute_name}"

    # --- PCA projection
    emb_2d = PCA(n_components=2).fit_transform(emb.cpu())

    # --- Plot by each level
    for level_idx, (label_tensor, label_names) in enumerate(
        zip(labels_per_level, label_names_per_level)
    ):
        plt.figure(figsize=(6, 5))
        for label_id in torch.unique(label_tensor):
            mask = label_tensor == label_id
            label_name = label_names[label_id.item()]
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=label_name, s=80)
        plt.title(f"{title_prefix} (Color: Level {level_idx})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
