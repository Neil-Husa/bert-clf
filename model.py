"""
Text classification
"""

import torch
from typing import Optional, Text, Tuple, List
from torch import Tensor
# from torch.autograd import Variable
from transformers import BertTokenizer, BertModel


class TextClassification(torch.nn.Module):

    def __init__(self, max_length: Optional[int] = None,
                 num_class: Optional[int] = None):
        super(TextClassification, self).__init__()
        self.max_length = max_length
        self.num_class = num_class
        self.bert_dim = 768
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.classifizer = torch.nn.Linear(self.bert_dim, num_class)

    def forward(self, input_ids: Optional[Tensor],
                attention_mask: Optional[Tensor],
                token_type_ids: Optional[Tensor]) -> Tensor:
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        return self.classifizer(outputs.pooler_output)

    def summary(self):
        print("Model Structure:", self, "\n\n")
        for name, param in self.named_parameters():
            print(f"Layer:{name} | Size:{param.size} | value:{param[:2]} \n")
