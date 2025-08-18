import torch
import torch.nn as nn
from transformers import DistilBertModel


class BloomBERT(nn.Module):
    def __init__(self, output_dim=6):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.attention = nn.Linear(self.bert.config.hidden_size, 1)

        # two layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # attention pooling
        attn_weights = torch.softmax(self.attention(hidden_states).squeeze(-1), dim=1)
        pooled_output = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)

        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits
