import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import pickle
from torch.utils.data import DataLoader, Dataset
from torch import optim
import numpy as np
from data_preprocessing import tools

label2id,id2label,num_labels=tools.load_schema()
num_label = num_labels+1
tokenizer=tools.get_tokenizer()

class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.Tensor(in_size + int(bias_x),out_size,in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]  
    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        
        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping

class myModel(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str, dropout_rate: float):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.roberta_encoder.resize_token_embeddings(len(tokenizer))
        self.lstm=torch.nn.LSTM(input_size=768,hidden_size=768, \
                        num_layers=1,batch_first=True, \
                        dropout=0.5,bidirectional=True)
        self.logits_layer=torch.nn.Linear(in_features=4*768, out_features=num_label)


    def forward(self, input_ids, input_mask, input_seg, is_training=False):
        bert_output = self.roberta_encoder(input_ids=input_ids, 
                                            attention_mask=input_mask, 
                                            token_type_ids=input_seg) 
        encoder_rep = bert_output[0]
        encoder_rep,_ = self.lstm(encoder_rep)
        
        batch_size, seq_len, hid_size = encoder_rep.size()

        start_extend = encoder_rep.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # end_extend = encoder_rep.unsqueeze(1).expand(-1, seq_len, -1, -1)
        end_extend = torch.transpose(start_extend,dim0=1,dim1=2)
        span_matrix = torch.cat([start_extend, end_extend], 3)
        span_logits = self.logits_layer(span_matrix)

        span_prob = torch.nn.functional.softmax(span_logits, dim=-1)

        if is_training:
            return span_logits
        else:
            return span_prob