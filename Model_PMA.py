import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
from attention import CBAMLayer
from transformers import BertConfig
import torch.nn.functional as F
from Model_CharBERT import CharBERTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharBertModel(nn.Module):
    """
    Definition of the CharBertModel we defined to classify Malicious URLs
    """

    def __init__(self):
        super(CharBertModel, self).__init__()
        config = BertConfig.from_pretrained('charbert-bert-wiki')
        self.bert = CharBERTModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)  # Add a dropout layer
        self.fc = nn.Linear(768, 2)
        self.hidden_size = 768
        self.fuse = nn.Conv1d(2 * self.hidden_size, self.hidden_size, kernel_size=1)

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]

        # add char level information
        char_ids = x[3]
        start_ids = x[4]
        end_ids = x[5]

        # CharBERTModel return outputs as a tuple
        # outputs =
        # (sequence_output, pooled_output, char_sequence_output, char_pooled_output) + char_encoder_outputs[1:]
        # we need to fuse the sequence_output and char_sequence_output from all hidden layers
        outputs = self.bert(char_input_ids=char_ids, start_ids=start_ids, end_ids=end_ids, input_ids=context,
                            attention_mask=mask,
                            token_type_ids=types, output_hidden_states=True)

        # (pooled_output, char_pooled_output)
        pooled = (outputs[1], outputs[3])

        #  Encoder returns:
        #  last-layer hidden state, (all hidden states_word), (all_hidden_states_char), (all attentions)
        #  tuple(torch.Size([16, 200, 768]),...)
        sequence_repr = outputs[0]
        char_sequence_repr = outputs[2]
        # fuse two channel
        fuse_output =()
        for x1, x2 in zip(sequence_repr, char_sequence_repr):
            x = torch.cat([x1, x2], dim=-1)
            # x torch.Size([16, 200, 768*2])
            # print("torch.cat"+str(x.shape))
            x = x.unsqueeze(1)
            # print("x.unsqueeze(1)"+str(x.shape))
            x1 = x.permute(0, 2, 1)
            # print("x.permute"+str(x1.shape))
            y = self.fuse(x1)
            # print("self.fuse(x1)"+str(y.shape))
            y = y.permute(0, 2, 1)
            # print("y.permute"+str(y.shape))
            # y torch.Size([16, 200, 768])
            y_output = y.squeeze(1)
            # print("y_output"+str(y_output.shape))
            fuse_output += (y_output,)

        pyramid = tuple(fuse_output)
        pyramid = torch.stack(pyramid, dim=0).permute(1, 0, 2, 3)
        # torch.Size([16, 12, 200, 768])

        model_cbam = CBAMLayer(channel=12).to(DEVICE)
        pos_pooled = model_cbam.forward(pyramid)
        # torch.Size([16, 12, 200, 768])

        pyramid_levels = [1, 2, 3, 4]  # Can be customized as needed
        output_feature_size = 768  # Output feature size

        # Initialize a list to store pyramid pooling results
        pooled_features = []

        for level in pyramid_levels:
            # Calculate the pooling window size for each level
            window_size = pos_pooled.size(1) // level

            # Use average pooling for each level
            # pooled_feature_tensor = F.avg_pool2d(pos_pooled.permute(0, 2, 3, 1), (1, window_size)).permute(0, 2, 3, 1)
            pooled_feature_tensor = F.avg_pool2d(pos_pooled.permute(0, 3, 2, 1), (1, window_size)).permute(0, 3, 2, 1)
            # torch.Size([16, 200, 768, 12])

            # Add the pooling results for each level to the list
            pooled_features.append(pooled_feature_tensor)

        # Concatenate the pyramid pooling results along the feature dimension
        concatenated_features = torch.cat(pooled_features, dim=1)

        # compress the features
        compressed_feature_tensor = torch.mean(concatenated_features, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)

        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)

        return pyramid, pooled, out


class Model(nn.Module):
    """
    Definition of the Basic Model we defined to classify Malicious URLs
    """

    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("charbert-bert-wiki")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)  # Add a dropout layer
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]

        # BertModel return encoded_layers, pooled_output
        # output_all_encoded_layers=True
        outputs, pooled = self.bert(input_ids=context, token_type_ids=types,
                                    attention_mask=mask,
                                    output_all_encoded_layers=True)

        pyramid = tuple(outputs)
        pyramid = torch.stack(pyramid, dim=0).permute(1, 0, 2, 3)
        # torch.Size([16, 12, 200, 768])

        model_cbam = CBAMLayer(channel=12).to(DEVICE)
        pos_pooled = model_cbam.forward(pyramid)
        # torch.Size([16, 12, 200, 768])

        pyramid_levels = [1, 2, 3, 4]  # Can be customized as needed
        output_feature_size = 768  # Output feature size

        # Initialize a list to store pyramid pooling results
        pooled_features = []

        for level in pyramid_levels:
            # Calculate the pooling window size for each level
            window_size = pos_pooled.size(1) // level

            # Use average pooling for each level
            # pooled_feature_tensor = F.avg_pool2d(pos_pooled.permute(0, 2, 3, 1), (1, window_size)).permute(0, 2, 3, 1)
            pooled_feature_tensor = F.avg_pool2d(pos_pooled.permute(0, 3, 2, 1), (1, window_size)).permute(0, 3, 2, 1)
            # torch.Size([16, 200, 768, 12])

            # Add the pooling results for each level to the list
            pooled_features.append(pooled_feature_tensor)

        # Concatenate the pyramid pooling results from different levels along the feature dimension
        concatenated_features = torch.cat(pooled_features, dim=1)

        # compress the features
        compressed_feature_tensor = torch.mean(concatenated_features, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)

        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)  # It is the result of classification

        return pyramid, pooled, out
