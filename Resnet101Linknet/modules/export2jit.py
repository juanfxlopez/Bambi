import torch
from resnet101inter_linknet_model import ResNetLinkModel
from helper import load_model
from torch import nn

"""
def save_model(cust_model, name="dense_segm.pt"):
    return torch.save(cust_model.module.state_dict(), name)


segm_model=ResNetLinkModel(input_channels=1,num_classes=2)
segm_model=nn.DataParallel(segm_model)
segm_model=load_model(segm_model, model_dir="./ResNet101inter_linknet_384_20_best.pt")
save_model(segm_model, name="./stuff.pt")
"""
segm_model=ResNetLinkModel(input_channels=1,num_classes=2)
segm_model=load_model(segm_model, model_dir="./stuff.pt")
dummyInput = torch.randn(1, 1, 384, 384)
tracedNet = torch.jit.trace(segm_model, example_inputs=dummyInput)
tracedNet.save("./jit_binaries/jit_resNet__384_20epcs.pt")