from collections import defaultdict

import torch


def augment_texts_with_labels(texts, labels, prefix="Attribute"):
    return [f"{prefix}: {label}. {text}" for text, label in zip(texts, labels)]


def build_hierarchy_config_from_paths(label_paths, attr_name):
    levels = list(zip(*label_paths))  # [L][N]
    num_levels = len(levels)

    level_mappings = []
    label_ids_per_level = []

    for level_labels in levels:
        unique = sorted(set(level_labels))
        mapping = {label: i for i, label in enumerate(unique)}
        level_mappings.append(mapping)
        label_ids_per_level.append([mapping[l] for l in level_labels])

    label_tensors = [torch.tensor(level) for level in label_ids_per_level]

    config = []
    for i in range(num_levels):
        entry = {
            "num_classes": len(level_mappings[i]),
            "label_mapping": level_mappings[i],
            "level_name": f"{attr_name}_level_{i}",
        }
        if i > 0:
            # Build mapping matrix from parent to current level
            parent_ids = label_ids_per_level[i - 1]
            child_ids = label_ids_per_level[i]
            parent_size = len(level_mappings[i - 1])
            child_size = len(level_mappings[i])
            M = torch.zeros(parent_size, child_size)

            for p, c in zip(parent_ids, child_ids):
                M[p, c] = 1.0

            entry["mapping"] = M  # <-- THIS is what the loss expects
        config.append(entry)
    return config, label_tensors
