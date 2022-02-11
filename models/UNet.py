
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import kornia.augmentation as K
import kornia as ko
import numpy as np
from models.BaseUNet import BaseUNet,up
from voxelmorph.torch.losses import Dice
import monai
import matplotlib.pyplot as plt
import kornia

class UNet(pl.LightningModule):
    # @property
    # def logger(self):
    #     return self._logger

    # @logger.setter
    # def logger(self, value):
    #     self._logger = value
    @property
    def automatic_optimization(self):
        return False
    def norm(self, x):
        # x = (x-torch.mean(x))/torch.std(x)
        # return x*
        if len(x.shape)==4:
            x = kornia.enhance.normalize_min_max(x)
        elif len(x.shape)==3:
            x= kornia.enhance.normalize_min_max(x[:, None, ...])[:,0, ...]
        else:
            x = kornia.enhance.normalize_min_max(x[None, None, ...])[0, 0, ...]
        return x

    def __init__(self,n_channels=1,n_classes=2,learning_rate=1e-4,weight_decay=1e-8,taa=False,selected_slices=None):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.taa=taa
        self.segmentor= BaseUNet(1,self.n_classes)
        #self.segmentor=monai.networks.nets.BasicUNet(2,1,self.n_classes)
        self.selected_slices=selected_slices
        print('lr',self.learning_rate)
        self.save_hyperparameters()
    def forward(self, x):
        sx=self.segmentor(x)
        return {'sx':sx}

    def spatial(self,x,y):
        trans = K.AugmentationSequential(K.RandomAffine(degrees=[-20,20], scale=[0.8,1.2],shear=[-20,20], resample="nearest", p=0.9), data_keys=["input", "mask"])
        x,y=trans(x,y)
        return x,y    

    def training_step(self, batch, batch_nb):
        X,Y=batch
        y_opt=self.optimizers()
        losses=[]
        for i in range(X.shape[2]):
            x=X[:,:,i,...]
            y=Y[:,:,i,...]
            if len(torch.unique(torch.argmax(y,1)))>1:
                y_opt.zero_grad()
                x,y=self.spatial(x, y)
                y_hat=self.forward(x)['sx']
                loss=nn.CrossEntropyLoss()(y_hat,torch.argmax(y,1))
                self.manual_backward(loss)
                y_opt.step()
                self.log('train_loss', loss)
                losses.append(loss)
        print(len(losses),torch.stack(losses).mean())
        self.log('val_accuracy',self.current_epoch)
        # return loss


    # def validation_step(self, batch, batch_idx):
    #     X, Y = batch
    #     out = self(x)
    #     y = y.cpu().detach()
    #     if self.n_classes > 1:
    #         out['sx'] = torch.argmax(out['sx'].cpu().detach(), 1, False)
    #     pred_oh = torch.moveaxis(
    #         F.one_hot(out['sx'].long(), self.n_classes), -1, 1)
    #     y_oh = torch.moveaxis(F.one_hot(y.long(), self.n_classes), -1, 1)
    #     dice_score = monai.metrics.compute_meandice(
    #         pred_oh, y_oh, include_background=True).cpu().detach()
    #     dice_score = torch.nan_to_num(dice_score)
    #     self.log('val_accuracy', dice_score.mean())
    #     for lab in range(dice_score.shape[-1]):
    #         self.log(f'val_accuracy_lab{lab}', dice_score[:, lab].mean())

    # def test_step(self, batch, batch_nb):
    #     x, y = batch
    #     x = x.to("cuda")
    #     y = y.to("cuda")

    #     y_hat = self(x)["sx"]

    #     # Kornia TAA
    #     if self.taa != False:
    #         y_pred = []
    #         y_pred.append(y_hat.cpu().detach())
    #         trans = K.AugmentationSequential(
    #             K.RandomAffine(degrees=4, scale=(0.95, 1.05), p=1), data_keys=["input"]
    #         )
    #         for i in range(10):
    #             x_t = trans(x)
    #             y_hat = self.net_model(x_t)["sx"]
    #             x_t.cpu().detach()
    #             y_inv = trans.inverse(y_hat)
    #             y_hat.cpu().detach()
    #             y_pred.append(y_inv.cpu().detach())
    #         y_pred = torch.stack(y_pred).mean(0)

    #     y_pred = torch.argmax(y_hat, 1, keepdim=False)
    #     accuracy = []
    #     y_pred = y_pred.to("cpu")
    #     y = y.to("cpu")
    #     pred_oh = torch.moveaxis(
    #         F.one_hot(y_pred.long(), self.n_classes), -1, 1)
    #     y_oh = torch.moveaxis(F.one_hot(y.long(), self.n_classes), -1, 1)
    #     for j in range(y.shape[0]):
    #         dsc = monai.metrics.compute_meandice(
    #             pred_oh[j: j + 1], y_oh[j: j + 1], include_background=False
    #         )
    #         dsc = torch.nan_to_num(dsc)
    #         accuracy.append(dsc.numpy())

    #     return {"test_accuracy_list": accuracy}

    # def test_epoch_end(self, outputs):
    #     accuracy = flatten([x["test_accuracy_list"] for x in outputs])
    #     fig = plt.figure()  # create a figure object
    #     accuracy = [list(x.flatten()) for x in accuracy]
    #     accuracies = dict()
    #     ax = fig.add_subplot()
    #     for lab in range(len(list(accuracy)[0])):
    #         accuracy_lab = [list(x)[lab] for x in list(accuracy)]
    #         print(accuracy_lab)
    #         ax.plot(range(len(accuracy)), accuracy_lab)
    #         mean = np.mean(accuracy_lab)
    #         self.log(f"test_accuracy_lab{lab}", mean)
    #         accuracies[f"test_accuracy_lab{lab}"] = mean
    #     return accuracies
    def register_images(self,x,x2,y):
        x=None
        y=None
        y_hat=self.forward(x2)['sx']
        return None,y_hat,None
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

def dice(res, gt, label): 
    A = gt == label
    B = res == label    
    TP = len(np.nonzero(A*B)[0])
    FN = len(np.nonzero(A*(~B))[0])
    FP = len(np.nonzero((~A)*B)[0])
    DICE = 0
    if (FP+2*TP+FN) != 0:
        DICE = float(2)*TP/(FP+2*TP+FN)
    return DICE*100

