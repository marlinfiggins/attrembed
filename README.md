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
