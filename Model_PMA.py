import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
from attention import CBAMLayer
import torch.nn.functional as F
from Model_CharBERT import CharBERTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharBertModel(nn.Module):
    """
    Definition of the CharBertModel we defined to classify Malicious URLs  
    """
    def __init__(self):
        super(CharBertModel, self).__init__()
        self.bert = CharBERTModel(BertModel.from_pretrained('charbert-bert-wiki'))
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)  # Add a dropout layer
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]
        outputs, pooled = self.bert(context, token_type_ids=types,
                                    attention_mask=mask,
                                    output_all_encoded_layers=True)

        pyramid = outputs[0].unsqueeze(0)
        pyramid = tuple(pyramid)
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
        outputs, pooled = self.bert(context, token_type_ids=types,
                                    attention_mask=mask,
                                    output_all_encoded_layers=True)

        pyramid = outputs[0].unsqueeze(0)
        pyramid = tuple(pyramid)
        pyramid = torch.stack(pyramid, dim=0).permute(1, 0, 2, 3)
        # torch.Size([16, 12, 200, 768])

        model_cbam = CBAMLayer(channel=12).to(DEVICE)
        pos_pooled = model_cbam.forward(pyramid)
        # torch.Size([16, 12, 200, 768])

        pyramid_levels = [1, 2, 3, 4]   # Can be customized as needed
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

