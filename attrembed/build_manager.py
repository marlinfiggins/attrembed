import torch
from attribute_manager import AttributeEmbeddingManager
from base_encoder import BaseEncoder
from helpers import build_hierarchy_config_from_paths


def build_manager_from_dataframe(
    df,
    flat_attributes=None,
    hierarchical_attributes=None,
    model_name=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    embedding_col="Vector",
    id_col="ID",
    text_col="Text",
    train_epochs=50
):
    """
    Build an AttributeEmbeddingManager from a dataframe with text and/or embeddings.
    
    If embeddings are not provided, model_name must be specified to compute them.

    Args:
        df: DataFrame with ID, text, attribute labels, and optionally embeddings.
        flat_attributes: List of flat attribute columns (single string labels).
        hierarchical_attributes: List of hierarchical attribute columns (list of strings).
        model_name: Transformer model name to use if computing embeddings.
        device: Torch device.
        embedding_col: Column name for base embedding vectors (if precomputed).
        id_col: Column name for unique IDs.
        text_col: Column name for text data.
        train_epochs: Number of training epochs. Set to 0 to skip classifier training.
    """
    ids = df[id_col].tolist()
    texts = df[text_col].tolist() if text_col in df.columns else None

    # Initialize encoder and manager
    if embedding_col in df.columns:
        encoder = BaseEncoder(model_name=model_name, device=device) if model_name else None
        manager = AttributeEmbeddingManager(encoder, device=device)
        manager.load_from_dataframe(df, embedding_col=embedding_col, id_col=id_col, text_col=text_col)
    elif model_name is not None:
        if texts is None:
            raise ValueError(f"Text column '{text_col}' not found in DataFrame.")
        encoder = BaseEncoder(model_name=model_name, device=device)
        manager = AttributeEmbeddingManager(encoder, device=device)
        manager.add_texts(ids, texts)
    else:
        raise ValueError(
            f"Neither precomputed embeddings ('{embedding_col}') nor a model_name was provided.\n"
            "Please supply one of the two to proceed."
        )

    # Identify attributes if not provided
    reserved_cols = {id_col, text_col, embedding_col}
    if flat_attributes is None:
        flat_attributes = [
            col for col in df.columns
            if col not in reserved_cols and df[col].apply(lambda x: isinstance(x, str)).all()
        ]
    if hierarchical_attributes is None:
        hierarchical_attributes = [
            col for col in df.columns
            if col not in reserved_cols and df[col].apply(lambda x: isinstance(x, list)).all()
        ]

    if train_epochs > 0:
        # Train flat classifiers
        for attr in flat_attributes:
            labels = df[attr].tolist()
            unique_labels = sorted(set(labels))
            label_to_id = {l: i for i, l in enumerate(unique_labels)}
            label_ids = [label_to_id[l] for l in labels]

            manager.train_classifier(attr, ids, label_ids, num_classes=len(unique_labels), epochs=train_epochs)
            manager.classifiers[attr].config = {
                "label_mapping": label_to_id,
                "id_to_label": {i: l for l, i in label_to_id.items()}
            }

        # Train hierarchical classifiers
        for attr in hierarchical_attributes:
            label_paths = df[attr].tolist()
            hierarchy_config, label_tensors = build_hierarchy_config_from_paths(label_paths, attr_name=attr)
            manager.train_hierarchical_classifier(attr, ids, label_tensors, hierarchy_config, epochs=train_epochs)

            for level_cfg in hierarchy_config:
                mapping = level_cfg["label_mapping"]
                level_cfg["id_to_label"] = {i: l for l, i in mapping.items()}
            manager.classifiers[attr].config = hierarchy_config

        # Compute attribute embeddings
        manager.compute_all_attributes(ids)

    return manager, ids, texts
