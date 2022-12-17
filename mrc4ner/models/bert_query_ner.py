#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_query_ner.py

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertPreTrainedModel,
    MegatronBertPreTrainedModel,
    MegatronBertModel,
)

from models.classifier import MultiNonLinearClassifier
from models.prefix_encoder import PrefixEncoder


class BertQueryNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQueryNER, self).__init__(config)
        self.bert = BertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(
            config.hidden_size * 2,
            1,
            config.mrc_dropout,
            intermediate_hidden_size=config.classifier_intermediate_hidden_size,
        )

        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """

        bert_outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(
            -1
        )  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(
            -1
        )  # [batch, seq_len, 1]

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits


class MegatronQueryNER(MegatronBertPreTrainedModel):
    def __init__(self, config):
        super(MegatronQueryNER, self).__init__(config)
        self.bert = MegatronBertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(
            config.hidden_size * 2,
            1,
            config.mrc_dropout,
            intermediate_hidden_size=config.classifier_intermediate_hidden_size,
        )

        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """

        bert_outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        _, seq_len, _ = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(
            -1
        )  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(
            -1
        )  # [batch, seq_len, 1]

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits


class BertPrefixQueryNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(
            config.hidden_size * 2,
            1,
            config.mrc_dropout,
            intermediate_hidden_size=config.classifier_intermediate_hidden_size,
        )

        self.hidden_size = config.hidden_size
        self.pre_seq_len = config.pre_seq_len
        
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        # self.prefix_tokens = torch.arange(self.pre_seq_len)
        self.prefix_encoder = PrefixEncoder(config)

        self.init_weights()

    def get_prompt(self, prefix_tokens):
        # prefix_tokens = (
        #     self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        # )
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.n_layer * 2, self.n_head, self.n_embd
        )
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, prompts=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(prompts)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(
            self.bert.device
        )
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        bert_outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, _ = sequence_heatmap.size()
        attention_mask = attention_mask[:, self.pre_seq_len :].contiguous()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(
            -1
        )  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(
            -1
        )  # [batch, seq_len, 1]

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits


class MegatronPrefixQueryNER(MegatronBertPreTrainedModel):
    def __init__(self, config):
        super(MegatronQueryNER, self).__init__(config)
        self.bert = MegatronBertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(
            config.hidden_size * 2,
            1,
            config.mrc_dropout,
            intermediate_hidden_size=config.classifier_intermediate_hidden_size,
        )

        self.hidden_size = config.hidden_size

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len)
        self.prefix_encoder = PrefixEncoder(config)

        self.init_weights()

    def get_prompt(self, batch_size):
        prefix_tokens = (
            self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        )
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size, self.pre_seq_len, self.n_layer * 2, self.n_head, self.n_embd
        )
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, past_key_values=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """

        batch_size = input_ids.shape[0]
        # past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(
            self.bert.device
        )
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        bert_outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        _, seq_len, _ = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(
            -1
        )  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(
            -1
        )  # [batch, seq_len, 1]

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits
