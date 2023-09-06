# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class QueryEncoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int = 512,
                 embed_dim: int = 300,
                 num_layers: int = 1,
                 bidirection: bool = True) -> None:
        super(QueryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=embed_dim,
            padding_idx=0)
        # self.embedding.weight.data.copy_(torch.load('glove_weights'))
        self.biLSTM = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
            batch_first=True,
            bidirectional=bidirection)

        self.W3 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.W2 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(3)])
        self.W1 = nn.Linear(hidden_dim * 2, 1)

    def extract_textual(self, q_encoding: Tensor, lstm_outputs: Tensor,
                        q_length: Tensor, t: int):
        q_cmd = self.W3(q_encoding).relu()
        q_cmd = self.W2[t](q_cmd)
        q_cmd = q_cmd[:, None, :] * lstm_outputs
        raw_att = self.W1(q_cmd).squeeze(-1)

        raw_att = apply_mask1d(raw_att, q_length)
        att = raw_att.softmax(dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def forward(self, query_tokens: Tensor,
                query_length: Tensor) -> List[Tensor]:
        self.biLSTM.flatten_parameters()

        query_embedding = self.embedding(query_tokens)

        # output denotes the forward and backward hidden states in Eq 2.
        query_embedding = pack_padded_sequence(
            query_embedding, query_length.cpu(), batch_first=True)
        output, _ = self.biLSTM(query_embedding)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # q_vector denotes the global representation `g` in Eq 2.
        q_vector_list = []

        for i, length in enumerate(query_length):
            h1 = output[i][0]
            hs = output[i][length - 1]
            q_vector = torch.cat((h1, hs), dim=-1)
            q_vector_list.append(q_vector)
        q_vector = torch.stack(q_vector_list)
        # outputs denotes the query feature in Eq3 in 3 levels.
        outputs = []
        for cmd_t in range(3):
            query_feat = self.extract_textual(q_vector, output, query_length,
                                              cmd_t)
            outputs.append(query_feat)

        # Note: the output here is zero-padded
        # we need slice the non-zero items for the following operations.
        return outputs


def apply_mask1d(attention: Tensor, image_locs: Tensor) -> Tensor:
    batch_size, num_loc = attention.size()
    tmp1 = torch.arange(
        num_loc, dtype=attention.dtype, device=attention.device)
    tmp1 = tmp1.expand(batch_size, num_loc)

    tmp2 = image_locs.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask = tmp1 >= tmp2.to(tmp1.dtype)
    attention = attention.masked_fill(mask, -1e30)
    return attention
