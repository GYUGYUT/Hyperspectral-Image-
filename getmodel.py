from torchvision import models
import torch.nn as nn
import torch
#from efficientnet_pytorch import EfficientNet


def getModel(backbone,numclass):
    net = None
    if backbone == 'resnet50': 
        net = models.resnet50(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                nn.Dropout(0.5),
                                nn.Linear(1024,numclass) )

    elif backbone == 'resnet101': 
        net = models.resnet101(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                nn.Dropout(0.5),
                                nn.Linear(1024,numclass) )
    elif backbone == 'resnet151': 
        net = models.resnet152(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                nn.Dropout(0.5),
                                nn.Linear(1024,numclass) )
    elif backbone == 'densenet121': 
        net = models.densenet121(weights="IMAGENET1K_V1")
        net.classifier = nn.Sequential( nn.Linear( 1024, 512),
                                nn.Dropout(0.5),
                                nn.Linear(512,numclass) )

    elif backbone == 'densenet169': 
        net = models.densenet169(weights="IMAGENET1K_V1")
        net.classifier = nn.Sequential( nn.Linear(1664, 832),
                                        nn.Dropout(0.5),
                                        nn.Linear(832,numclass) )

    elif backbone == 'vgg16' : 
        net = models.vgg16(weights="IMAGENET1K_V1")
        net.classifier[6] = nn.Sequential( nn.Linear(4096, 2048),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048,numclass) )
    elif backbone == 'vgg19' : 
        net = models.vgg19(weights="IMAGENET1K_V1")
        net.classifier[6] = nn.Sequential( nn.Linear(4096, 2048),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048,numclass) )
    elif backbone == 'alexnet' : 
        net = models.alexnet(weights="IMAGENET1K_V1")
        net.classifier[6] = nn.Sequential( nn.Linear(4096, 2048),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048,numclass) )

    elif backbone == 'resnext50' : 
        net = models.resnext50_32x4d(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024,numclass) )
    elif backbone == 'resnext101' : 
        net = models.resnext101_32x8d(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024,numclass) )
    elif backbone == 'shufflenet' : 
        net = models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear( 1024, 512),
                                nn.Dropout(0.5),
                                nn.Linear(512,numclass) )
    elif backbone == 'mobilenet_v2' : 
        net = models.mobilenet_v2(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 1280, 640),
                                nn.Dropout(0.5),
                                nn.Linear(640,numclass) )
    elif backbone == 'mobilenet_v3' : 
        net = models.MobileNetV3(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 1280, 640),
                                nn.Dropout(0.5),
                                nn.Linear(640,numclass) )
    elif backbone == 'mnasnet' : 
        net = models.mnasnet1_0(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 1280, 640),
                                nn.Dropout(0.5),
                                nn.Linear(640,numclass) )
    elif backbone == 'efficientnet_b6' : 
        net = models.efficientnet_b6(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 2560 , 1280),
                                nn.Dropout(0.5),
                                nn.Linear(1280,numclass) )
    elif backbone == 'efficientnet_b7' : 
        net = models.efficientnet_b7(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 2560 , 1280),
                                nn.Dropout(0.5),
                                nn.Linear(1280,numclass) )
    elif backbone == 'regnet_y_32gf' : 
        net = models.regnet_y_32gf()
        net.fc= nn.Sequential( nn.Linear( 3712 , 1856),
                                nn.Dropout(0.5),
                                nn.Linear(1856,numclass) )

    # class Nine_Channel(nn.Module):
    #     def __init__(self):
    #         super(Nine_Channel, self).__init__()
    #         # 입력 이미지 채널을 9채널로 조정하는 부분 수정
    #         self.nine_ch = nn.Conv2d(9, 3, kernel_size=1, stride=1)
    #         # ResNet50 모델 불러오기
    #         self.net = net

    #     def forward(self, x):
    #         x = self.nine_ch(x)
    #         x = self.net(x)
    #         return x
    # net = Nine_Channel()
    print(net)
    return net