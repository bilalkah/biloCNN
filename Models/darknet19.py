import torch
import torch.nn as nn

net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            #[(1024, 3), (1024, 3)],
            # conv4
            #[(1024, 3)],
            [(1000,1)]
        ]

class Darknet19(nn.Module):
    def __init__(self,num_classes=1000):
        super(Darknet19,self).__init__()
        self.features = self.create_darknet19_features(num_classes)
        self.fc = nn.Linear(1000,num_classes)
    def create_darknet19_features(self,num_classes):
        features = []
        in_channels = 6
        for _,layers in enumerate(net_cfgs):
            for layer in layers:
                if layer == 'M':
                    features.append(nn.MaxPool2d(kernel_size=(2,2),stride=2))
                else:
                    out_channels,kernel_size = layer
                    if kernel_size == 1:
                        features.append(nn.Conv2d(in_channels,out_channels,kernel_size,1))
                    else:
                        features.append(nn.Conv2d(in_channels,out_channels,kernel_size,1,1))
                    features.append(nn.BatchNorm2d(out_channels))
                    features.append(nn.LeakyReLU(0.1))
                    in_channels = out_channels
        features.append(nn.AdaptiveAvgPool2d((1,1)))
        return nn.Sequential(*features)  

    
    def forward(self,x):
        x = self.features(x).view(x.size(0),-1)
        return self.fc(x)

if __name__ == '__main__':
    model = Darknet19(num_classes=10)
    x=torch.randn(4,6,224,224)
    print(model(x).shape)
    # print(model)
    

        