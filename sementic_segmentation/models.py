import torch
from torch import nn


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()
        
        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60
        
        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
            n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1
        )
        self.feature_res_2 = ResidualBlockPSP(
            n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1
        )
        self.feature_dilation_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2
        )
        self.feature_dilation_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4
        )
        
        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_size=[6, 3, 2, 1], height=img_size_8, width=img_size_8)
        
        self.decode_feature = DecodePSPFeature(
            height=img_size, width=img_size, n_classes=n_classes
        )
        
        self.aux = AuxiliaryPSPlayers(
            in_channels=1024, height=img_size, width=img_size, n_classes=n_classes
        )
        
    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilation_res_1(x)
        
        output_aux = self.aux(x)
        
        x = self.feature_dilation_res_2(x)
        
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)
        
        return (output, output_aux)
    
    
class ConvBlockWithActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(ConvBlockWithActivation, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs
    
    
class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()
        
        self.conv1 = ConvBlockWithActivation(3, 64, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.conv2 = ConvBlockWithActivation(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv3 = ConvBlockWithActivation(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        outputs = self.maxpool(x)
        return outputs
    
    
class ResidualBlockPSP(nn.Sequnetial):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()
        
        self.add_module('block1', BottleneckPSP(in_channels, mid_channels, out_channels, stride, dilation))
        
        for i in range(n_blocks - 1):
            self.add_module('block' + str(i+2), BottleneckPSP(out_channels, mid_channels, stride, dilation))
            

class ConvBlockWithoutActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(ConvBlockWithoutActivation, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)
        return outputs
    

class BottleneckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(BottleneckPSP, self).__init__()
        
        self.conv1 = ConvBlockWithActivation(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.conv2 = ConvBlockWithActivation(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.conv3 = ConvBlockWithoutActivation(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        
        self.conv_residual = ConvBlockWithoutActivation(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        conv = self.conv3(self.conv2(self.conv1(x)))
        residual = self.conv_residual(x)
        return self.relu(conv + residual)
        