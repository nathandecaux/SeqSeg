
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import kornia.augmentation as K
import kornia as ko
import numpy as np
from visualisation.plotter import norm, plot_results, flatten
from models.BaseUNet import BaseUNet,up
import monai
import matplotlib.pyplot as plt
import kornia
from kornia.geometry import ImageRegistrator
from kornia.geometry.transform import elastic_transform2d
from voxelmorph.torch.networks import VxmDense
from voxelmorph.torch.layers import SpatialTransformer
from models.TorchIR.torchir.networks.dirnet import DIRNet
from models.TorchIR.torchir.networks.globalnet import AIRNet
from models.TorchIR.torchir.transformers import BsplineTransformer, DiffeomorphicFlowTransformer,AffineTransformer
# from models.TorchIR.torchir.metrics import NCC
from voxelmorph.torch.losses import NCC,Grad,Dice



class ScanRegister(pl.LightningModule):
    # @property
    # def logger(self):
    #     return self._logger

    # @logger.setter
    # def logger(self, value):
    #     self._logger = value
    @property
    def automatic_optimization(self):
        if self.dim==2:
            return True
        else:
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
    

    def __init__(self,n_channels=1,n_classes=2,learning_rate=9e-4,weight_decay=1e-8,taa=False,ckpt=None,way='up',dim=3,shape=256):
        super().__init__()
        self.n_classes = n_classes
        print(n_classes)
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.taa=taa
        self.sigmas=nn.Parameter(torch.ones(2))
        self.alpha=nn.Parameter(torch.ones(2))
        self.dim=dim
        size=shape
        if self.dim==2:
            shape=(size,size)
        else:
            shape=(80,size,size)
        self.multi_class_hp=nn.Parameter(torch.ones(self.n_classes)/(self.n_classes))
        self.registrator= VxmDense(shape,bidir=False,int_downsize=1,int_steps=7)
        self.grid_spacing=(8,8)
        self.way=way
        self.metric=NCC()
        self.grad_loss=Grad()
        #self.registrator =  DIRNet(self.grid_spacing)
        # self.registrator = AIRNet(kernels=16)
        # self.bspline_transformer = AffineTransformer(ndim=2)
        #self.registrator.requires_grad=True
        #self.bspline_transformer = BsplineTransformer(ndim=2, upsampling_factors=self.grid_spacing)

        self.lab_transformer=SpatialTransformer(shape,'nearest')
        self.save_hyperparameters()

    def bspline_transformer(self,trans,x2,x):
        return self.registrator.transformer(x,trans)

        

    def forward(self, X,X2,registration=False):            
        X_hat,trans=self.registrator.forward(X,X2,registration=registration)            
        return X_hat,trans
    
    def compute_loss(self,x1_hat,x2,y=None,y2=None,trans=None):
        loss=NCC().loss(x1_hat,x2)
        y_hat=self.registrator.transformer(y,trans)
        for i in range(y2.shape[2]):
            if len(torch.unique(torch.argmax(y2,1)))>1:
                loss+=Dice().loss(y_hat[:,:,i,...],y2[:,:,i,...])
        return loss

    def blend(self,x,y):
        x=self.norm(x)
        blended=torch.stack([y,x,x])
        return blended

    def training_step(self, batch, batch_nb):
        if self.dim==3:
            X,Y,X2,Y2=batch
            y_opt=self.optimizers()
            y_opt.zero_grad()
            X1_hat,trans=self.forward(X,X2)
            loss=self.compute_loss(X1_hat,X2,Y,Y2,trans)
            self.manual_backward(loss)
            y_opt.step()
            self.log_dict({'loss':loss},prog_bar=True)
            return loss
        else:
            X,Y,X2=batch
            y_opt=self.optimizers()
            for i in range(X.shape[2]):
                x=self.X[:,:,i,...]
                x2=self.X[:,:,i,...]

                y_opt.zero_grad()
                x_hat,trans=self.forward(x,x2)
                loss=self.compute_loss(x_hat,x2)
                self.log_dict({'loss':loss},prog_bar=True)
                self.manual_backward(loss)
                y_opt.step()

    def validation_step(self,batch,batch_nb):
        X,Y,X2,Y2=batch
        X_hat,trans=self.forward(X,X2,registration=True)
        Y_hat=self.registrator.transformer(Y,trans)

        dices=[]
        for i in range(Y2.shape[2]):
            y2=Y2[:,:,i,...]
            y_hat=Y_hat[:,:,i,...]
            if len(torch.unique(torch.argmax(y2,1)))>1:
                dice=monai.metrics.compute_meandice(
                y_hat>0.5, y2, include_background=False)
                dices.append(dice)        
        dice_score = torch.nan_to_num(torch.stack(dices))
        self.log('val_accuracy', dice_score.mean())
        print(dice_score.mean())
        return dice_score.mean()

    def register_images(self,x1,x2,y1):
        x1_hat,trans=self.forward(x1,x2,registration=True)
        return self.registrator.transformer(x1,trans),self.bspline_transformer(trans,y1,y1),trans
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,amsgrad=True)
    def on_after_backward(self):
        global_step = self.global_step
        for name, param in self.registrator.named_parameters():
            # self.logger.experiment.add_histogram(name, param, global_step)
            if param.requires_grad:
                self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)
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

