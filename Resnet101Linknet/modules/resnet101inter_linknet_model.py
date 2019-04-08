from torchvision import models
from torch import nn
import torch


class ConvEluGrNorm(nn.Module):
    def __init__(self, inp_chnl, out_chnl):
        super(ConvEluGrNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=inp_chnl, out_channels=out_chnl, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=16, num_channels=out_chnl)
        self.elu = nn.CELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.elu(out)
        return out

class UpsampleLayer(nn.Sequential):
    def __init__(self, in_chnl, mid_chnl, out_chnl, transp=False):
        super(UpsampleLayer, self).__init__()
        if not transp:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvEluGrNorm(in_chnl, mid_chnl),
                ConvEluGrNorm(mid_chnl, out_chnl)
            )
        else:
            self.block = nn.Sequential(
                ConvEluGrNorm(in_chnl, mid_chnl),
                nn.ConvTranspose2d(in_channels=mid_chnl, out_channels=out_chnl, 
                        kernel_size=4, stride=2, padding=1, bias=False),
                nn.CELU(inplace=True)
            )

class Resnet101SegmModel(nn.Module):
    def __init__(self, input_channels, num_filters=32, num_classes=1, pretrained=False):
        super(Resnet101SegmModel, self).__init__()
        net_model = models.resnet101(pretrained=pretrained)
        modules = list(net_model.children())[:-1]
        encoder=nn.Sequential(*modules)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False), # torch.Size([1, 64, 224, 224])
            nn.GroupNorm(num_groups=16, num_channels=64), #starting with image size: 128x128
            nn.CELU(inplace=True), #torch.Size([1, 64, 256, 256])
            encoder[3] #torch.Size([1, 64, 64, 64])
        )                             
        self.layer2 = encoder[4]#torch.Size([1, 256, 64, 64]) 
        self.layer3 = encoder[5]#torch.Size([1, 512, 32, 32]) -> torch.Size([1, 512+x, 32, 32])   
        self.layer4 = encoder[6]#torch.Size([1, 1024, 16, 16]) -> torch.Size([1, 1024+x, 16, 16])  
        self.layer5 = encoder[7]#torch.Size([1, 2048, 8, 8])-> torch.Size([1, 2048, 8, 8])
        self.layer6 = encoder[8]#torch.Size([1, 2048, 2, 2)

        self.dec6 = nn.Sequential (
            UpsampleLayer(in_chnl=2048, mid_chnl=num_filters*8, out_chnl=num_filters*8),  # dec([1, 2048, 2, 2]) -> ([1, 2048, 4, 4])
            UpsampleLayer(in_chnl=256, mid_chnl=num_filters*8, out_chnl=num_filters*8)  # dec([1, 2048, 4, 4]) -> ([1, 2048, 8, 8])
        )
        self.dec5 = UpsampleLayer(2048 + num_filters*8, num_filters*8, num_filters*8)#dec6([1,2048 + 256,8,8]) = ([1,256,16,16])
        self.dec4 = UpsampleLayer(1024 + num_filters*8, num_filters*8, num_filters*8)#dec6([1,1024 + 256,16,16]) = ([1,256,32,32])
        self.dec3 = UpsampleLayer(512 + num_filters*8, num_filters*8, num_filters*8)#dec4([1,512 + 256,32,32]) = ([1,256,64,64])
        self.dec2 = nn.Sequential (
                    ConvEluGrNorm(256 + num_filters*8, num_filters*8), #dec2([1,64 + 256,64,64]) = ([1,256,64,64])
                    ConvEluGrNorm(num_filters*8, num_filters*2) #dec2([1,256,64,64]) = ([1,64,64,64])
                )
        self.dec1 =  nn.Sequential (
                    UpsampleLayer(64 + num_filters*2, num_filters*2, num_filters*2),  #dec1([1,64 + 64,64,64]) = ([1,64,128,128])
                    UpsampleLayer(num_filters*2, num_filters, num_filters)         #dec1([1,64 + 64,128,128]) = ([1,32,256,256])
                )
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)  #conv2d([1,32,256,256])= ([1,1,256,256])

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        conv5 = self.layer5(conv4)
        out = self.layer6(conv5)

        dec6= self.dec6(out)
        dec5 = self.dec5(torch.cat([dec6,conv5],1))
        dec4 = self.dec4(torch.cat([dec5,conv4],1))
        dec3 = self.dec3(torch.cat([dec4,conv3],1))
        dec2 = self.dec2(torch.cat([dec3,conv2],1))
        dec1 = self.dec1(torch.cat([dec2,conv1],1))
        return self.final(dec1)

def ResNetLinkModel(input_channels, pretrained=False, num_classes=1):
    return Resnet101SegmModel(input_channels=input_channels, pretrained=pretrained, num_classes=num_classes)
