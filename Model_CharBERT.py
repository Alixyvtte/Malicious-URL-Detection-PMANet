import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_utils import BertLayer, ACT2FN, BertPooler

""" CharBERT model from the paper 'CharBERT: Character-aware Pre-trained Language Model' """

BertLayerNorm = torch.nn.LayerNorm

class CharBERTModel(nn.Module):
    def __init__(self, config, is_roberta=False):
        super(CharBERTModel, self).__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.char_embeddings = CharBertEmbeddings(config, is_roberta=is_roberta)

        self.encoder = CharBertEncoder(config, is_roberta=is_roberta)

        self.pooler = BertPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, char_input_ids=None, start_ids=None, end_ids=None, input_ids=None, attention_mask=None, \
                token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, \
                encoder_attention_mask=None):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.

        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
                                                                                                        attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape,
                        encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, \
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        char_embeddings = self.char_embeddings(char_input_ids, start_ids, end_ids)
        char_encoder_outputs = self.encoder(char_embeddings,
                                            embedding_output,
                                            attention_mask=extended_attention_mask,
                                            head_mask=head_mask,
                                            encoder_hidden_states=encoder_hidden_states,
                                            encoder_attention_mask=encoder_extended_attention_mask)

        sequence_output, char_sequence_output = char_encoder_outputs[0], char_encoder_outputs[1]
        pooled_output = self.pooler(sequence_output)
        char_pooled_output = self.pooler(char_sequence_output)

        outputs = (sequence_output, pooled_output, char_sequence_output, char_pooled_output) + char_encoder_outputs[
                                                                                               1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (all_hidden_states_for_word_and_char), (attentions)

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)


        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings  # + char_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

bert_charemb_config = {"char_vocab_size": 1001, \
                       "char_embedding_size": 256, \
                       "kernel_size": 5}
class CharBertEmbeddings(nn.Module):
    def __init__(self, config, is_roberta=False):
        super(CharBertEmbeddings, self).__init__()
        self.config = config
        self.char_emb_config = bert_charemb_config
        self.char_embeddings = nn.Embedding(self.char_emb_config["char_vocab_size"], \
                                            self.char_emb_config["char_embedding_size"], padding_idx=0)
        self.rnn_layer = nn.GRU(input_size=self.char_emb_config["char_embedding_size"], \
                                hidden_size=int(config.hidden_size / 4), batch_first=True, bidirectional=True)

    def forward(self, char_input_ids, start_ids, end_ids):
        input_shape = char_input_ids.size()
        # print(f"shape of char_input_ids in CharBertEmbeddings: {list(input_shape)}")
        assert len(input_shape) == 2

        batch_size, char_maxlen = input_shape[0], input_shape[1]

        char_input_ids_reshape = torch.reshape(char_input_ids, (batch_size, char_maxlen))
        char_embeddings = self.char_embeddings(char_input_ids_reshape)

        self.rnn_layer.flatten_parameters()
        all_hiddens, last_hidden = self.rnn_layer(char_embeddings)


        start_one_hot = nn.functional.one_hot(start_ids, num_classes=char_maxlen)

        end_one_hot = nn.functional.one_hot(end_ids, num_classes=char_maxlen)

        start_hidden = torch.matmul(start_one_hot.float(), all_hiddens)

        end_hidden = torch.matmul(end_one_hot.float(), all_hiddens)

        char_embeddings_repr = torch.cat([start_hidden, end_hidden], dim=-1)

        return char_embeddings_repr

class CharBertEncoder(nn.Module):
    """
     We make changes here, for outputting Both the all hidden states_word and  all_hidden_states_char
    """
    def __init__(self, config, is_roberta=False):
        super(CharBertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.word_linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # self.word_linear2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.char_linear1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # self.char_linear2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # self.share_bias = nn.Parameter(torch.zeros(config.hidden_size))
        if not is_roberta:
            fusion_layer = torch.nn.Conv1d(in_channels=config.hidden_size * 2, out_channels=config.hidden_size,
                                           kernel_size=3, padding=3 // 2)
            self.fusion_layer_list = nn.ModuleList([fusion_layer for _ in range(config.num_hidden_layers)])
        else:
            self.fusion_layer = torch.nn.Conv1d(in_channels=config.hidden_size * 2, out_channels=config.hidden_size,
                                                kernel_size=3, padding=3 // 2)

        self.act_layer = ACT2FN[config.hidden_act]
        self.word_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.char_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.is_roberta = is_roberta

    def forward(self, char_hidden_states, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states_char = ()
        all_hidden_states_word = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            fusion_layer = None
            if not self.is_roberta:
                fusion_layer = self.fusion_layer_list[i]
            else:
                fusion_layer = self.fusion_layer
            if self.output_hidden_states:
                all_hidden_states_word = all_hidden_states_word + (hidden_states,)
                all_hidden_states_char = all_hidden_states_char + (char_hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask)
            char_layer_outputs = layer_module(char_hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                              encoder_attention_mask)

            word_outputs = layer_outputs[0]
            char_outputs = char_layer_outputs[0]
            word_transform = self.word_linear1(word_outputs)
            char_transform = self.char_linear1(char_outputs)
            # share_hidden  = self.act_layer(word_transform + char_transform + self.share_bias)
            share_cat = torch.cat([word_transform, char_transform], dim=-1)
            share_permute = share_cat.permute(0, 2, 1)
            share_fusion = fusion_layer(share_permute)
            share_hidden = share_fusion.permute(0, 2, 1)

            hidden_states = self.word_norm(share_hidden + word_outputs)
            char_hidden_states = self.char_norm(share_hidden + char_outputs)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states_word = all_hidden_states_word + (hidden_states,)
            all_hidden_states_char = all_hidden_states_char + (char_hidden_states,)

        outputs = (hidden_states, char_hidden_states)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states_word,) + (all_hidden_states_char,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states_word), (all_hidden_states_char), (all attentions)
