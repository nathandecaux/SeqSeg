from tkinter import Image
import sys
# sys.path.append('/home/nathan/SeqSeg/models/TorchIR')
from pytorch_lightning import Trainer
from datetime import datetime
from kornia.geometry import ImageRegistrator
from soupsieve import select
from models.LabelProp import LabelProp
import torch
import numpy as np
import nibabel as ni
import kornia.geometry as KG
import torch.nn.functional as F
from copy import deepcopy
from voxelmorph.torch.networks import VxmDense
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import monai
from data.CleanDataModule import PlexDataModule
from data.DMDDataModule import DMDDataModule
from copy import copy,deepcopy
max_epochs=200

def to_batch(x,device='cpu'):
    return x[None,None,...].to(device)

def hardmax(Y,dim):
    return torch.moveaxis(F.one_hot(torch.argmax(Y,dim)), -1, dim)

def remove_annotations(Y,selected_slices):
    if selected_slices!=None:
        for i in range(Y.shape[1]):
            if i not in selected_slices:
                Y[:,i,...]=Y[:,i,...]*0
    return Y

def create_dict(keys,values):
    new_dict=dict()
    for k,v in zip(keys,values):
        new_dict[k]=v
    return new_dict

def get_weights(Y):
    flag=False
    weights=torch.zeros((Y.shape[1]))
    n=0
    for i in range(Y.shape[1]):
        if len(torch.unique(torch.argmax(Y[:,i,...],0)))>1:
            if not flag: flag=True
            else: 
                weights[i-(n):i]=weights[i-(n):i]-n/2
                weights[i]=0
                n=1
        else:
            if flag:
                weights[i]=n
                n+=1
    return (torch.arctan(weights)/3.14+0.5)


def propagate_labels(X,Y,model,model_down=None):
    Y2=deepcopy(Y)
    model.eval().to('cuda')
    if model_down==None: model_down=model
    else: model_down.eval()
    X=X[0]
    for i,x1 in enumerate(X):
        try:
            x2=X[i+1]
        except:
            print('End of volume')
        else:
            y1=Y[:,i,...]
            if len(torch.unique(torch.argmax(Y[:,i+1,...],0)))==1 and len(torch.unique(torch.argmax(y1,0)))>1:
                _,y,_=model.register_images(to_batch(x1,'cuda'),to_batch(x2,'cuda'),y1[None,...].to('cuda'))
                Y[:,i+1,...]=y.cpu().detach()[0]
    for i in range(X.shape[0]-1,1,-1):
        x1=X[i]
        try:
            x2=X[i-1]
        except:
            print('End of volume')
        else:
            y1=Y2[:,i,...]
            if len(torch.unique(torch.argmax(y1,0)))>1 and len(torch.unique(torch.argmax(Y2[:,i-1,...],0)))==1:
                _,y,_=model_down.register_images(to_batch(x1,'cuda'),to_batch(x2,'cuda'),y1[None,...].to('cuda'))
                Y2[:,i-1,...]=y.cpu().detach()[0]
    return Y,Y2

def compute_metrics(y_pred,y):
    dice=monai.metrics.compute_meandice(y_pred, y, include_background=False)
    hauss=monai.metrics.compute_hausdorff_distance(y_pred, y, include_background=False)
    asd=monai.metrics.compute_average_surface_distance(y_pred, y, include_background=False)
    return dice,hauss,asd

def get_dices(Y_dense,Y,Y2,selected_slices):
    weights=get_weights(remove_annotations(deepcopy(Y_dense),selected_slices))
    dices_up=[]
    hauss_up=[]
    asd_up=[]
    dices_down=[]
    hauss_down=[]
    asd_down=[]
    dices_sum=[]
    hauss_sum=[]
    asd_sum=[]
    dices_weighted=[]
    hauss_weighted=[]
    asd_weighted=[]
    for i in range(Y.shape[1]):
        if i not in selected_slices and len(torch.unique(torch.argmax(Y_dense[:,i,...],0)))>1:
            d_up,h_up,a_up=compute_metrics(hardmax(Y[:,i,...],0)[None,...], Y_dense[:,i,...][None,...])
            d_down,h_down,a_down=compute_metrics(hardmax(Y2[:,i,...],0)[None,...], Y_dense[:,i,...][None,...])
            d_sum,h_sum,a_sum=compute_metrics(hardmax(Y[:,i,...]+Y2[:,i,...],0)[None,...], Y_dense[:,i,...][None,...])
            d_weighted,h_weighted,a_weighted=compute_metrics(hardmax((1-weights[i])*Y[:,i,...]+weights[i]*Y2[:,i,...],0)[None,...], Y_dense[:,i,...][None,...])
            dices_up.append(d_up)
            dices_down.append(d_down)
            dices_sum.append(d_sum)
            dices_weighted.append(d_weighted)
            hauss_up.append(h_up)
            hauss_down.append(h_down)
            hauss_sum.append(h_sum)
            hauss_weighted.append(h_weighted)
            asd_up.append(a_up)
            asd_down.append(a_down)
            asd_sum.append(a_sum)
            asd_weighted.append(a_weighted)

    dice_up=torch.nan_to_num(torch.stack(dices_up)).mean().numpy()
    dice_down=torch.nan_to_num(torch.stack(dices_down)).mean().numpy()
    dice_sum=torch.nan_to_num(torch.stack(dices_sum)).mean().numpy()
    dice_weighted=torch.nan_to_num(torch.stack(dices_weighted)).mean().numpy()
    hauss_up=torch.nan_to_num(torch.stack(hauss_up)).mean().numpy()
    hauss_down=torch.nan_to_num(torch.stack(hauss_down)).mean().numpy()
    hauss_sum=torch.nan_to_num(torch.stack(hauss_sum)).mean().numpy()
    hauss_weighted=torch.nan_to_num(torch.stack(hauss_weighted)).mean().numpy()
    asd_up=torch.nan_to_num(torch.stack(asd_up)).mean().numpy()
    asd_down=torch.nan_to_num(torch.stack(asd_down)).mean().numpy()
    asd_sum=torch.nan_to_num(torch.stack(asd_sum)).mean().numpy()
    asd_weighted=torch.nan_to_num(torch.stack(asd_weighted)).mean().numpy()
    dices=create_dict(['dice_up','dice_down','dice_sum','dice_weighted','hauss_up','hauss_down','hauss_sum','hauss_weighted','asd_up','asd_down','asd_sum','asd_weighted'],[dice_up,dice_down,dice_sum,dice_weighted,hauss_up,hauss_down,hauss_sum,hauss_weighted,asd_up,asd_down,asd_sum,asd_weighted])
    return dices

def get_test_data(data_PARAMS):
    data_PARAMS=deepcopy(data_PARAMS)
    dataset=data_PARAMS.pop('dataset')
    if dataset=='PLEX':
        dm=PlexDataModule(**data_PARAMS)
    else:
        dm=DMDDataModule(**data_PARAMS)
    dm.setup('fit')
    X,Y_dense=dm.val_dataloader().dataset[0]
    return X,Y_dense

def train_and_eval(data_PARAMS,model_PARAMS,ckpt=None):
    data_PARAMS=deepcopy(data_PARAMS)
    model_PARAMS=deepcopy(model_PARAMS)
    dataset=data_PARAMS.pop('dataset')
    if dataset=='PLEX':
        dm=PlexDataModule(**data_PARAMS)
        model_PARAMS['n_classes']=2
    else:
        dm=DMDDataModule(**data_PARAMS)
        model_PARAMS['n_classes']=11

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m%Y-%H%M%S")
    dir=f'/home/nathan/SeqSeg/voxelmorph_ckpts/labelprop/bench'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath=f'{dir}/bench',
        filename='labelprop-{epoch:02d}-{val_accuracy:.2f}-'+dt_string,
        save_top_k=1,
        mode='max',
    )
    logger=TensorBoardLogger("bench_logs", name="label_prop",log_graph=True)
    model=LabelProp(**model_PARAMS)
    if ckpt!=None:
        model=model.load_from_checkpoint(ckpt,strict=False)
    else:
        trainer=Trainer(gpus=1,max_epochs=max_epochs,logger=logger,callbacks=checkpoint_callback)
        trainer.fit(model,dm)
        model=model.load_from_checkpoint(checkpoint_callback.best_model_path)
    dm.setup('fit')
    _,Y_dense=dm.val_dataloader().dataset[0]
    dm.setup('test')
    X,Y=dm.test_dataloader().dataset[0]
    Y=remove_annotations(Y,data_PARAMS['selected_slices']['000'])
    Y_up,Y_down=propagate_labels(X,Y,model)
    res=get_dices(Y_dense,Y_up,Y_down,data_PARAMS['selected_slices']['000'])
    res['ckpt']=checkpoint_callback.best_model_path if ckpt==None else ckpt
    return Y_up,Y_down,res

