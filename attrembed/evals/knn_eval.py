from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def compute_knn_confusion_matrix_from_manager(
    manager, ids, labels, attribute=None, k=5, return_preds=False
):
    id_to_label = {i: labels[idx].item() for idx, i in enumerate(ids)}

    def get_embedding(i):
        return (
            manager.get_embedding(i, attribute)
            if attribute
            else manager.base_embeddings[i]
        )

    preds = []
    trues = []

    for query_id in ids:
        query_emb = get_embedding(query_id)
        rest_ids = [i for i in ids if i != query_id]
        rest_embs = torch.stack([get_embedding(i) for i in rest_ids])

        sims = F.cosine_similarity(query_emb.unsqueeze(0), rest_embs)
        topk = torch.topk(sims, k=k)
        neighbor_labels = [id_to_label[rest_ids[i]] for i in topk.indices]
        majority = Counter(neighbor_labels).most_common(1)[0][0]

        preds.append(majority)
        trues.append(id_to_label[query_id])

    cm = confusion_matrix(trues, preds, labels=sorted(set(trues)))

    if return_preds:
        return cm, trues, preds
    return cm


def plot_confusion_matrix(cm, labels, title="kNN Retrieval Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compare_knn_confusion_spaces(
    manager, ids, labels, label_names, attribute_name, level_name="", k=10
):
    cm_base = compute_knn_confusion_matrix_from_manager(
        manager, ids, labels, attribute=None, k=k
    )
    plot_confusion_matrix(
        cm_base, labels=label_names, title=f"{level_name} – Base Embedding kNN"
    )

    cm_attr = compute_knn_confusion_matrix_from_manager(
        manager, ids, labels, attribute=attribute_name, k=k
    )
    plot_confusion_matrix(
        cm_attr,
        labels=label_names,
        title=f"{level_name} – {attribute_name} Embedding kNN",
    )

    return cm_base, cm_attr
