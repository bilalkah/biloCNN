import torch
import torch.nn as nn

"""
    t = expansion factor
    c = number of output channel
    n = repeat 
    s = stride
    t   c   n   s
"""
MobileNetV1_arch = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

class DepthSepConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(DepthSepConvBlock,self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=stride,padding=1,groups=in_channels,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
        
        

class BottleneckBlock(nn.Module):
    def __init__(self,in_channels,out_channels,expand_ratio,stride):
        super(BottleneckBlock,self).__init__()
        self.stride = stride
        self.expand_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels*expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels*expand_ratio),
            nn.ReLU6(inplace=True),
        )
        self.depthwise_separable_layer = DepthSepConvBlock(
            in_channels=in_channels*expand_ratio,
            out_channels=out_channels,
            stride=stride,
        )
    
    def forward(self,x):
        out = self.expand_layer(x)
        out = self.depthwise_separable_layer(out)
        if self.stride==1 and x.shape[1:] == out.shape[1:]:
            return x + out
        return out

class MobileNetV2(nn.Module):
    def __init__(self,num_classes=10):
        super(MobileNetV2,self).__init__()
        self.first_conv_channels = 32
        self.last_conv_channels = 1280
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = self.first_conv_channels,
                kernel_size = 3,
                stride = 2,
                padding = 1
            ),
            nn.BatchNorm2d(self.first_conv_channels),
        )
        
        self.bottleneck_blocks = self.make_bottleneck(MobileNetV1_arch)
        
        self.last_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=320,
                out_channels=1280,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AvgPool2d(7),
            nn.Conv2d(
                in_channels=1280,
                out_channels=num_classes,
                kernel_size=1,
            )
        )
        
        
    def make_bottleneck(self,arch):
        in_channels = 32
        bottleneck = []
        for _,(t,c,n,s) in enumerate(arch):
            for i in range(n):
                if i == 1:
                    s=1
                if i+1 != n:
                    bottleneck.append(BottleneckBlock(in_channels,in_channels,t,s))
                else:    
                    bottleneck.append(BottleneckBlock(in_channels,c,t,s))
            in_channels = c
        return nn.Sequential(*bottleneck)
    
    def forward(self,x):
        x = self.initial_conv(x)
        x = self.bottleneck_blocks(x)
        x = self.last_layers(x).view(x.shape[0],-1)
        return x
        

if __name__ == '__main__':
    net = MobileNetV2()
    x = torch.randn(6,3,224,224)
    print(net(x).shape)