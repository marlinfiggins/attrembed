## attrembed

**attrembed** is a library for generating compact, interpretable, and attribute-aware embeddings from text. It allows you to compress general-purpose transformer embeddings into specialized subspaces aligned with metadata attributes like topic, format, or rhetorical function.

These compressed embeddings are especially useful for:

- Retrieval based on a specific attribute (e.g., topic-aware similarity)
- Visualization and analysis of attribute-driven structure in text data
- Attribute-conditioned clustering, classification, or filtering
- Combining multiple attribute spaces using learned or manual weighting

attrembed supports both **flat** and **hierarchical** attributes and includes tools for training, exporting, evaluation, and visualization.

---

## Installation

```bash
git clone https://github.com/marlinfiggins/attrembed.git
cd attrembed
poetry install
poetry shell
```

## Usage Overview

attrembed supports two workflows:

### Starting from Text + Attribute Labels Only

You provide a DataFrame with
- An ID column
- A text column
- One or more attribute columns

Example DataFrame:

| ID | Text                                           | topic_class | reading_level  |
|----|------------------------------------------------|-------------|----------------|
| 0  | Photosynthesis converts sunlight into energy. | Biology     | introductory   |
| 1  | The trolley problem illustrates moral tradeoffs. | Philosophy  | intermediate   |

We can then load this data frame, compute frozen base embeddings from the model, train classifiers for each attribute, and store a normalized logit embedding per attribute with the following code

```python
from attrembed import build_manager_from_dataframe

manager, ids, texts = build_manager_from_dataframe(
    df=df,
    flat_attributes=["topic_class", "reading_level"],
    hierarchical_attributes=[],
    model_name="bert-base-uncased",  # or any HF model
    id_col="ID",
    text_col="Text"
)
```

### Starting from Precomputed Embeddings

If you already have base embeddings stored in a column, provide a DataFrame like:

| ID | Text                                        | Vector            | topic_class |
|----|---------------------------------------------|-------------------|-------------|
| 0  | Quantum mechanics deals with uncertainty.   | [0.1, 0.2, …]     | Physics     |
| 1  | Genes transmit hereditary information.      | [0.3, 0.4, …]     | Biology     |

Here, `Vector` is a list of floats (e.g., 768-dim). Loading the dataframe including the embeddings, training the classifiers, and storing the embeddings can be done with:

```python
manager, ids, texts = build_manager_from_dataframe(
    df=df,
    flat_attributes=["topic_class"],
    hierarchical_attributes=[],
    embedding_col="Vector",
    id_col="ID",
    text_col="Text"
)
```

### Evaluation on New Data

To evaluate on a test set:

1. Load the test set into the manager using `load_from_dataframe`
2. Compute attribute-specific embeddings using `compute_all_attributes`
3. Evaluate using cosine-similarity-based retrieval, kNN classification, or similarity analysis

An example using the kNN classification is given below:

```python
test_ids = test_df["ID"].tolist()
manager.load_from_dataframe(test_df, embedding_col="Vector")
manager.compute_all_attributes(test_ids)

from attrembed.evaluation.knn import compute_knn_confusion_matrix_from_manager

label_map = manager.classifiers["topic_class"].config["label_mapping"]
label_ids = [label_map[l] for l in test_df["topic_class"]]

cm = compute_knn_confusion_matrix_from_manager(
    manager, test_ids, label_ids, attribute="topic_class", k=5
)

```

### Exporting for Reuse

Once trained, you can export everything you need into a single file:

```python
from attrembed.utils.export import export_combined_parquet

export_combined_parquet(
    dataset=your_dataset_list,
    manager=manager,
    ids=ids,
    file_path="combined_data.parquet"
)
```

This saves a .parquet file with columns: `ID`, `Text`, `Vector` (base embedding), and Attribute labels (flat and hierarchical).

Supported Attribute Types:
- Flat: Single-level categorical labels (e.g. `reading_level`)
- Hierarchical: Multi-level labels with tree-like structure (e.g., ["Science", "Biology", "Genetics"])

You can mix both in the same dataset.
