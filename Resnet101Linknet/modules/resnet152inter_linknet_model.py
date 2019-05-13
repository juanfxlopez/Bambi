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

class Resnet152SegmModel(nn.Module):
    def __init__(self, input_channels, num_filters=32, num_classes=1, pretrained=False):
        super(Resnet152SegmModel, self).__init__()
        net_model = models.resnet152(pretrained=pretrained)
        modules = list(net_model.children())[:-1]
        encoder=nn.Sequential(*modules)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False), # torch.Size([1, 64, 384, 384])
            nn.GroupNorm(num_groups=16, num_channels=64), #starting with image size: 384x384
            nn.CELU(inplace=True), #torch.Size([1, 64, 384, 384])
            encoder[3] #torch.Size([1, 64, 96, 96])
        )                             
        self.layer2 = encoder[4]#torch.Size([1, 256, 96, 96]) 
        self.layer3 = encoder[5]#torch.Size([1, 512, 48, 48]) -> torch.Size([1, 512+x, 48, 48])   
        self.layer4 = encoder[6]#torch.Size([1, 1024, 24, 24]) -> torch.Size([1, 1024+x, 24, 24])  
        self.layer5 = encoder[7]#torch.Size([1, 2048, 12, 12])-> torch.Size([1, 2048, 12, 12])
        #self.layer6 = encoder[8]#torch.Size([1, 2048, 6, 6)  ,, used for  image with original size =256
        self.layer6 = nn.AvgPool2d(kernel_size=9, stride=2, padding=0)#torch.Size([1, 2048, 3, 3) for image with original size =512

        self.dec6 = nn.Sequential (
            UpsampleLayer(in_chnl=2048, mid_chnl=num_filters*8, out_chnl=num_filters*8),  # dec([1, 2048, 3, 3]) -> ([1, 2048, 6, 6])
            UpsampleLayer(in_chnl=256, mid_chnl=num_filters*8, out_chnl=num_filters*8)  # dec([1, 2048, 6, 6]) -> ([1, 2048, 12, 12])
        )
        self.dec5 = UpsampleLayer(2048 + num_filters*8, num_filters*8, num_filters*8)#dec6([1,2048 + 256,12,12]) = ([1,256,24,24])
        self.dec4 = UpsampleLayer(1024 + num_filters*8, num_filters*8, num_filters*8)#dec6([1,1024 + 256,24,24]) = ([1,256,48,48])
        self.dec3 = UpsampleLayer(512 + num_filters*8, num_filters*8, num_filters*8)#dec4([1,512 + 256,48,48]) = ([1,256,96,96])
        self.dec2 = nn.Sequential (
                    ConvEluGrNorm(256 + num_filters*8, num_filters*8), #dec2([1,64 + 256,96,96]) = ([1,256,96,96])
                    ConvEluGrNorm(num_filters*8, num_filters*2) #dec2([1,256,96,96]) = ([1,64,96,96])
                )
        self.dec1 =  nn.Sequential (
                    UpsampleLayer(64 + num_filters*2, num_filters*2, num_filters*2),  #dec1([1,64 + 64,96,96]) = ([1,64,192,192])
                    UpsampleLayer(num_filters*2, num_filters, num_filters)         #dec1([1,64 + 64,192,192]) = ([1,32,384,384])
                )
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)  #conv2d([1,32,384,384])= ([1,1,384,384])

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
    return Resnet152SegmModel(input_channels=input_channels, pretrained=pretrained, num_classes=num_classes)
