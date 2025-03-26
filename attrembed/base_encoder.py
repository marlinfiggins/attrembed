from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch

class BaseEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", device="cpu", pooling="cls"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.device = device
        self.pooling = pooling
        self.encoder.to(device)

    def encode(self, texts, batch_size=32, max_length=256):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                    max_length=max_length).to(self.device)
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                if self.pooling == "cls":
                    pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
                elif self.pooling == "mean":
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size())
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, dim=1)
                    sum_mask = mask.sum(1)
                    pooled = sum_embeddings / sum_mask
                else:
                    raise ValueError(f"Unknown pooling type: {self.pooling}")
                all_embeddings.append(pooled.cpu())
        return torch.cat(all_embeddings, dim=0)
