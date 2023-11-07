import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
from attention import CBAMLayer
import torch.nn.functional as F
from Model_CharBERT import CharBertModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharBertModel(nn.Module):
    def __init__(self):
        super(CharBertModel, self).__init__()
        self.bert = CharBertModel(BertModel.from_pretrained('charbert-bert-wiki'))
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)  # 添加dropout层
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

        pyramid_levels = [1, 2, 3, 4]  # 可以根据需要自定义级别
        output_feature_size = 768  # 输出特征的大小

        # 初始化用于存储金字塔池化结果的列表
        pooled_features = []

        for level in pyramid_levels:
            # 计算每个级别的池化窗口大小
            window_size = pos_pooled.size(1) // level

            # 使用平均池化对每个级别进行池化操作
            pooled_feature_tensor = F.avg_pool2d(pos_pooled.permute(0, 3, 2, 1), (1, window_size)).permute(0, 3, 2, 1)
            # torch.Size([16, 768, 200,12])

            # 将每个级别的池化结果添加到列表中
            pooled_features.append(pooled_feature_tensor)

        # 将不同级别的金字塔池化结果在特征维度上拼接起来
        concatenated_features = torch.cat(pooled_features, dim=1)
        # torch.Size([16, 10, 200, 768])

        # 最终压缩特征为 torch.Size([16, 768])
        compressed_feature_tensor = torch.mean(concatenated_features, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)

        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)  # 是分类的结果

        return pyramid, pooled, out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("/hy-tmp/urls/charbert-bert-wiki")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.1)  # 添加dropout层
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

        pyramid_levels = [1, 2, 3, 4]  # 可以根据需要自定义级别
        output_feature_size = 768  # 输出特征的大小

        # 初始化用于存储金字塔池化结果的列表
        pooled_features = []

        for level in pyramid_levels:
            # 计算每个级别的池化窗口大小
            window_size = pos_pooled.size(1) // level

            # 使用平均池化对每个级别进行池化操作
            pooled_feature_tensor = F.avg_pool2d(pos_pooled.permute(0, 3, 2, 1), (1, window_size)).permute(0, 3, 2, 1)
            # torch.Size([16, 768, 200,12])

            # 将每个级别的池化结果添加到列表中
            pooled_features.append(pooled_feature_tensor)

        # 将不同级别的金字塔池化结果在特征维度上拼接起来
        concatenated_features = torch.cat(pooled_features, dim=1)
        # torch.Size([16, 10, 200, 768])

        # 最终压缩特征为 torch.Size([16, 768])
        compressed_feature_tensor = torch.mean(concatenated_features, dim=2)
        compressed_feature_tensor = torch.mean(compressed_feature_tensor, dim=1)

        out = self.dropout(compressed_feature_tensor)
        out = self.fc(out)  # 是分类的结果

        return pyramid, pooled, out
