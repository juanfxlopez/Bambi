import os
import torch
from torch import optim

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def save_model(cust_model, name="dense_segm.pt"):
    return torch.save(cust_model.state_dict(), name)

def load_model(cust_model, model_dir="dense_segm.pt", map_location_device="cpu"):
    if map_location_device == "cpu":
        cust_model.load_state_dict(torch.load(model_dir, map_location=map_location_device))
    elif map_location_device == "gpu":
        cust_model.load_state_dict(torch.load(model_dir))
    cust_model.eval()
    return cust_model

def save_checkpoint(model, optimizer,epoch,epoch_loss,aver_jaccard,aver_jaccard_inter):
    checkpoint = { 'Epochs': epoch,
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch_loss': epoch_loss,
        'aver_jaccard': aver_jaccard,
        'aver_jaccard_inter': aver_jaccard_inter}
    checkpoint_name= "checkpoint_epoch" + str(epoch) + ".pth"
    torch.save(checkpoint, checkpoint_name)
    print(checkpoint_name)
    if (epoch>1):
        checkpoint_previousname= "checkpoint_epoch" + str(epoch-1) + ".pth"
        os.remove(checkpoint_previousname)

def load_checkpoint(model,filepath):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        checkpoint = torch.load(filepath)
        model.cuda()
    else:
        checkpoint = torch.load(filepath ,map_location="cpu")
    
    '''for parameter in model.parameters():
        #parameter.requires_grad = False
        print (parameter.requires_grad)'''

    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    phase='train'
    print("| {} Loss: {:.4f} | Jaccard Average Acc: {:.4f} | Jaccard Average Acc inter: {:.4f} |".format(phase, checkpoint['epoch_loss'], checkpoint['aver_jaccard'],checkpoint['aver_jaccard_inter']))
    print("_"*15)
    #model.eval()
    #model.train()
    return model,optimizer
