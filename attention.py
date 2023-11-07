import torch.nn as nn

class CBAMLayer(nn.Module):
    """
        Initializes a CBAM (Attention Module) Layer.

        :param channel: The number of input channels.
        :param reduction: The reduction ratio for channel attention (default is 3).
        :param spatial_kernel: The size of the spatial kernel (default is 7).

        It's worth noting that we only utilize the channel attention in CBAM.
    """
    def __init__(self, channel, reduction=3, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # Channel attention: Compress H and W to 1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            # Use Conv2d for more convenient operations compared to Linear
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),  # Inplace operation to save memory
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        return x
