import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, n_classes, dropout_rate):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # here im gonna freeze all the layers
        for param in self.ber.parameters():
            param.requires_grad = False

        # and here im gonna unfreeze the last few layers
        for param in self.bert.encoder.layer[-3:].parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(dropout_rate)

        self.classdifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, n_classes)
            )

        self.attention = nn.Linear(self.bert.config.hidded_size, 1)


    def attention_pooling(self, hidden_states, attention_mask):
        '''
        hidden_states.shape >>> (B,S,H) S:sequence_length, H:gidden_size
        attetion_mask.shape >>> (B,S)
        '''
        attention_scores = self.attention(hidden_states).squeeze(-1) #compute the linear transformation for each token in the hidden_state / (B,S)
        attention_scores = attention_scores_masked_fill(attention_mask == 0, float('-inf')) #basically applying the mask / (B,S)
        attention_probs = torch.softmax(attention_scores, dim=-1) #compute attention probabilities / (B,S)
        pooled_output = torch.bmm(attnetion_probs.unsequeeze(1), hidden_states).sequeeze(-1) #compute weighted-sum / (B,1,S)@(B,S,H).sequeeze(-1) >>> (B,H)
        return pooled_output

    def forward(self, input_ids, attention_mask):
        '''
        input_ids.shapee >>> (B,S)
        attention_mask.shape >>> (B,S)
        '''
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask) #obtain the BERT output 
        hidden_states = outputs.last_hidden_state #(B,S,H)

        pooled_output = self.attention_pooling(hidden_states, attention_mask) #(B,H)
        
        pooled_output = self.deopout(pooled_output) #(B,H)

        logits = self.classifier(pooled_output) #(B,C) C:number of classes

        return logits
