import sys
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
import monai
import matplotlib.pyplot as plt
import kornia
from kornia.geometry import ImageRegistrator
from kornia.geometry.transform import elastic_transform2d
from voxelmorph.torch.networks import VxmDense
from voxelmorph.torch.layers import SpatialTransformer
from voxelmorph.torch.losses import NCC,Grad,Dice
from easyreg.voxel_morph import PixelMorph
from easyreg.mermaid_net import MermaidNet
from mermaid.module_parameters import ParameterDict as MermaidParameterDict
import json
from models.voxelmorph2D import VoxelMorph2d
class LabelProp(pl.LightningModule):

    @property
    def automatic_optimization(self):
        if self.dim==2:
            return True
        else:
            return False
    def norm(self, x):
        
        if len(x.shape)==4:
            x = kornia.enhance.normalize_min_max(x)
        elif len(x.shape)==3:
            x= kornia.enhance.normalize_min_max(x[:, None, ...])[:,0, ...]
        else:
            x = kornia.enhance.normalize_min_max(x[None, None, ...])[0, 0, ...]
        return x
    

    def __init__(self,n_channels=1,n_classes=2,learning_rate=1e-3,weight_decay=1e-8,taa=False,ckpt=None,way='up',dim=3,size=256,selected_slices=None,losses={}):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.taa=taa
        self.sigmas=nn.Parameter(torch.ones(2))
        self.alpha=nn.Parameter(torch.ones(2))
        self.selected_slices=selected_slices
        if isinstance(size,int):size=[size,size]
        self.bidir=False
        if way=='both' : self.bidir=True
        self.registrator= VxmDense(size,bidir=self.bidir,int_downsize=1,int_steps=7)
        self.nearest_transformer=SpatialTransformer(size,mode='bilinear')
        self.way=way
        self.metric=NCC()
        self.dim=dim
        self.losses=losses
        self.save_hyperparameters()

    def apply_trans(self,x,trans,mode='bilinear'):
        return self.registrator.transformer(x,trans)
        
    def forward(self, x,x2,y=None,registration=False):            
        if self.bidir:
            out=self.registrator.forward(x,x2,registration)  
            if registration:
                x1_hat,trans,neg_trans=out
                return x1_hat,trans,neg_trans
            else:
                x1_hat,_,trans=out
                return x1_hat,trans,-trans

        else:
            x1_hat,trans=self.registrator.forward(x,x2,registration=registration)      
            return x1_hat,trans,None
      
    def compute_loss(self,x1_hat=None,x2=None,y=None,y2=None,trans=None):
        loss_ncc=0
        loss_seg=0
        loss_trans=0
        if x1_hat!=None:
            loss_ncc=NCC().loss(x1_hat,x2)
        if y!=None:
            loss_seg= Dice().loss(y,y2)
        if trans!=None:
            loss_trans=Grad().loss(trans,trans)
        return loss_ncc+loss_seg+loss_trans

    def blend(self,x,y):
        x=self.norm(x)
        blended=torch.stack([y,x,x])
        return blended

    def training_step(self, batch, batch_nb):
        X,Y=batch
        y_opt=self.optimizers()
        pred=None
        pred_x=None
        loss=[]
        y_opt.zero_grad()
        chunks=[]
        chunk=[]
        loss_up=[]
        loss_down=[]
        for i in range(X.shape[2]):
            y2=Y[:,:,i,...]
            if len(torch.unique(torch.argmax(y2,1)))>1:
                chunk.append(i)
            if len(chunk)==2:
                chunks.append(chunk)
                chunk=[i]
        if self.current_epoch==0:
            print(chunks)
        for chunk in chunks:
            y_opt.zero_grad()
            pos_seq=[]
            neg_seq=[]
            for i in range(chunk[0],chunk[1]):
                x=X[:,:,i,...]
                x2=X[:,:,i+1,...]
                if not self.way=='down':
                    x1_hat_f,pos,neg=self.forward(x,x2)
                    loss_up.append(self.compute_loss(x1_hat_f,x2,trans=pos))
                    pos_seq.append(pos)
                else:
                    x2_hat_b,neg,_=self.forward(x2,x)#

                if not self.way=='up':
                    neg_seq.append(neg)
                    x2_hat_b=self.registrator.transformer(x2,neg)
                    loss_down.append(self.compute_loss(x2_hat_b,x,trans=neg))
    
            #Better with mean
            # print('losses',loss_up,loss_down)
            if self.way=='up':
                loss=torch.stack(loss_up).mean()
            elif self.way=='down':
                loss=torch.stack(loss_down).mean()
            else:
                loss_up=torch.stack(loss_up).mean()
                loss_down=torch.stack(loss_down).mean()
                loss=(loss_up+loss_down)
            
            if not self.way=='down':
                pos_x=X[:,:,chunk[0]:chunk[0]+1,...]
                pos_y=Y[:,:,chunk[0]:chunk[0]+1,...]
                for pos in pos_seq:
                    pos_x=torch.cat((pos_x,self.apply_trans(pos_x[:,:,-1,...],pos).unsqueeze(2)),2)
                    pos_y=torch.cat((pos_y,(self.apply_trans(pos_y[:,:,-1,...],pos,'nearest').unsqueeze(2))),2)
                
                if self.losses['compo-reg-up']:
                    loss+=self.compute_loss(pos_x[:,:,-1,...],X[:,:,chunk[1],...])
                if self.losses['compo-dice-up']:
                    loss+=self.compute_loss(y=pos_y[:,:,-1,...],y2=Y[:,:,chunk[1],...])

            if not self.way=='up':
                neg_x=X[:,:,chunk[1]:chunk[1]+1,...]
                neg_y=Y[:,:,chunk[1]:chunk[1]+1,...]
                for neg in reversed(neg_seq):
                    neg_x=torch.cat((self.apply_trans(neg_x[:,:,0,...],neg).unsqueeze(2),neg_x),2)
                    neg_y=torch.cat(((self.apply_trans(neg_y[:,:,0,...],neg,'nearest').unsqueeze(2)),neg_y),2)
                if self.losses['compo-reg-down']:
                    loss+=self.compute_loss(neg_x[:,:,0,...],X[:,:,chunk[0],...])
                if self.losses['compo-dice-down']:
                    loss+=self.compute_loss(y=neg_y[:,:,0,...],y2=Y[:,:,chunk[0],...])

            if self.way=='both':
                #This helps
                if self.losses['bidir-cons-dice']:
                    loss+=self.compute_loss(y=neg_y,y2=pos_y)
                #This breaks stuff
                if self.losses['bidir-cons-reg']:
                    loss+=self.compute_loss(pos_x,neg_x)

            self.log_dict({'loss':loss},prog_bar=True)
            self.manual_backward(loss)
            y_opt.step()
            loss_up=[]
            loss_down=[]
            # self.logger.experiment.add_image('x_true',X[0,:,chunk[0],...])
            # self.logger.experiment.add_image('neg_x',neg_x[0,:,0,...])
            # self.logger.experiment.add_image('x_true_f',X[0,:,chunk[1],...])
            # self.logger.experiment.add_image('pos_x',pos_x[0,:,-1,...])
        return loss

    def validation_step(self,batch,batch_nb):
        X,Y=batch
        X2=deepcopy(X)
        if self.selected_slices!=None:
            Y_dense=deepcopy(Y)
            for i in range(Y.shape[2]):
                if i not in self.selected_slices:
                    Y[:,:,i,...]=Y[:,:,i,...]*0  
        Y_true=torch.clone(Y)
        Y2=torch.clone(Y)
        dices=[]
        dices_chunk=[]
        dices_dense=[]
        dices_dense_down=[]
        chunks=[]
        chunk=[]
        for i in range(X.shape[2]):
            y2=Y[:,:,i,...]
            if len(torch.unique(torch.argmax(y2,1)))>1:
                chunk.append(i)
            if len(chunk)==2:
                chunks.append(chunk)
                chunk=[i]
        for chunk in chunks:
            pos_seq=[]
            neg_seq=[]
            n=chunk[1]-chunk[0]
            weights=[]
            for k,i in enumerate(range(chunk[0],chunk[1])):
                x=X[:,:,i,...]
                x2=X[:,:,i+1,...]
                x1_hat,pos,_=self.forward(x,x2,registration=True)
                x2_hat,neg,_=self.forward(x2,x,registration=True)
                pos_seq.append(pos)
                neg_seq.append(neg)
                weights.append(torch.ones(1)*k-n/2)
            weights.append(torch.ones(1)*n-n/2)
            weights=torch.stack(weights)
            weights=(torch.arctan(weights)/3.14+0.5).cuda()
            for i,pos in enumerate(pos_seq):
                pos_y=self.apply_trans((Y[:,:,chunk[0]+i,...]),pos,'nearest')
                Y[:,:,chunk[0]+i+1,...]=pos_y
            
            for i,neg in enumerate(reversed(neg_seq)):
                neg_y=self.apply_trans((Y2[:,:,chunk[1]-i,...]),neg,'nearest')
                # neg_x=self.apply_trans(X2[:,:,chunk[1]-i,...],neg,'nearest')
                Y2[:,:,chunk[1]-i-1,...]=neg_y
                # X2[:,:,chunk[1]-i-1,...]=neg_x

            if not self.way=='down':
                dices_chunk.append(monai.metrics.compute_meandice(
                    self.hardmax(Y[:,:,chunk[1],...],1), Y_dense[:,:,chunk[1],...], include_background=False))
            if not self.way=='up':
                dices_chunk.append(monai.metrics.compute_meandice(
                    self.hardmax(Y2[:,:,chunk[0],...],1), Y_dense[:,:,chunk[0],...], include_background=False))
                Y=Y2
            # for i,w in enumerate(weights):
            #     Y[:,:,chunk[0]+i,...]*=1-w
            #     Y2[:,:,chunk[0]+i,...]*=w
            # print(dices_chunk)
            # neg_x=X_true[:,:,chunk[1],...]
            # neg_y=Y_true[:,:,chunk[1],...]
            # for neg in reversed(neg_seq):
            #     neg_x=self.apply_trans(neg_x,neg)
            #     neg_y=self.apply_trans(neg_y,neg)

        
        # if self.selected_slices!=None:
        #     for i in range(X.shape[2]):
        #         if i not in self.selected_slices and len(torch.unique(torch.argmax(Y[:,:,i,...],1)))>1:
        #             dice=monai.metrics.compute_meandice(
        #         self.hardmax(Y[:,:,i,...],1), Y_dense[:,:,i,...], include_background=False)
        #             dice_down=monai.metrics.compute_meandice(
        #         self.hardmax(Y2[:,:,i,...],1), Y_dense[:,:,i,...], include_background=False)
        #             dices_dense.append(dice)
        #             dices_dense_down.append(dice_down)
        #     print(dices_dense)
        #     dices_dense=torch.nan_to_num(torch.stack(dices_dense))
        #     dices_dense_down=torch.nan_to_num(torch.stack(dices_dense_down))

        #     print('dices dense (up / down)',dices_dense.mean(),dices_dense_down.mean())

        dices_chunk = torch.nan_to_num(torch.stack(dices_chunk))
        self.log('val_accuracy', dices_chunk.mean())
        print('dices chunk',dices_chunk.mean())
        return dices_chunk.mean()

    def register_images(self,x1,x2,y1):
        x1_hat,trans,_=self.forward(x1,x2,registration=True)
        x2_hat,neg,_=self.forward(x2,x1,registration=True)
        return self.apply_trans(x1,trans),self.apply_trans(y1,trans),neg
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,amsgrad=True)
    def on_after_backward(self):
        global_step = self.global_step
        for name, param in self.registrator.named_parameters():
            # self.logger.experiment.add_histogram(name, param, global_step)
            if param.requires_grad:
                self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)
    def hardmax(self,Y,dim):
        return torch.moveaxis(F.one_hot(torch.argmax(Y,dim)), -1, dim)
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

