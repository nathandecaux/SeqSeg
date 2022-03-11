import sys
from unittest import result
# sys.path.append('/home/nathan/SeqSeg/models/TorchIR')
from pytorch_lightning import Trainer
from copy import copy,deepcopy
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
import json
import pandas as pd
from train import *
def new_nii(X,type='float32'):
    affine=np.eye(4)
    if dataset=='PLEX':
        affine[0,0]=5/3
    else:
        affine[0,0]=5
    return ni.Nifti1Image(np.array(X).astype(type),affine)

def is_already_done(file,res):
    if len(file)==0:
        return False
    for result in file:
        if result['losses']==res['losses'] and result['way']==res['way']:
            return True
    return False
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(results,f):
    with open('results_unet.json','w') as f:
        json.dump(results,f, cls=NumpyArrayEncoder)


size=(256,256)

dataset='PLEX'

data_PARAMS = {'dataset':dataset,'batch_size':1,'subject_ids': [0], 'val_ids': [0], 'test_ids': [0],'aug':True,'dim':3,'shape':size}
model_PARAMS= {'model_name':'UNet'}



try:
    with open('results_unet.json') as f:
        results= json.load(f)
except:
    results=dict()

if dataset not in results.keys():
    results[dataset]=dict()
slices=dict()
for r in [5]:
    ratio=str(r)
    #selected_slices=[107, 199, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220]
    selected_slices=[107,153,199]#[107, 199, 0, 40, 80, 120, 160, 200]#[107,153,199]
    # if r%2==0:
    #     selected_slices+=list(range(224))[::r]
    # else:
    #     selected_slices+=list(range(224))[::r]
   
    slices[r]=selected_slices
    if ratio not in results[dataset].keys():
        print('noon')
        results[dataset][ratio]=[]
    model_PARAMS['selected_slices']=selected_slices
    data_PARAMS['selected_slices']={'000':selected_slices}
    res=dict()
    preds=dict()
    Y_up,Y_down,res_train=train_and_eval(data_PARAMS,model_PARAMS,ckpt="/home/nathan/SeqSeg/voxelmorph_ckpts/labelprop/bench/bench/labelprop-epoch=199-val_accuracy=199.00-14022022-104802.ckpt")#"/home/nathan/SeqSeg/voxelmorph_ckpts/labelprop/bench/bench/labelprop-epoch=199-val_accuracy=199.00-22022022-141950.ckpt")
    print(Y_up.shape)
    Y_fused=torch.argmax((1-res_train['weights']).T*torch.moveaxis(Y_up,1,-1)+(res_train['weights']).T*torch.moveaxis(Y_down,1,-1),0)
    Y_fused=new_nii(torch.moveaxis(Y_fused,-1,0).cpu().detach().numpy()[107:200],'uint8')
    ni.save(Y_fused,'Y_UNet.nii.gz')


    
    # res.update(res_train)
    # results[dataset][ratio].append(deepcopy(res))
    # save_json(results,f)


# ckpt_up=None#'labelprop-up-epoch=150-val_accuracy=0.50[1].ckpt'#"labelprop-up-epoch=92-val_accuracy=0.92[1].ckpt"#'labelprop-up-epoch=139-val_accuracy=1.00[0, 1].ckpt'#'labelprop-up-epoch=145-val_accuracy=0.98[0, 1].ckpt'
# resume_ckpt=None#'labelprop-up-epoch=145-val_accuracy=0.98[0, 1].ckpt'
# ckpt_down='labelprop-up-epoch=150-val_accuracy=0.50[1].ckpt'#'labelprop-up-epoch=97-val_accuracy=0.84[1].ckpt'
# selected_slices=[107,160,199]#+list(range(101,185))[::2]#+[156]+[185]
# data_PARAMS = {'batch_size':1,'subject_ids': [0], 'val_ids': [0], 'test_ids': [0],'aug':True,'dim':dim,'shape':size,'lab':1,'selected_slices':{'000':selected_slices,'002':None}}
# for r,v in slices.items():
#     print(r,len(np.unique(np.array(v))))
#     print(r,v)
# for r in [2,3,5,10,20]:
#     chunks=[]
#     chunk=[]
#     annot=0
#     for i in range(21,68):
#         if i in slices[r]:
#             chunk.append(i)
#             annot+=1
#         if len(chunk)==2:
#             chunks.append(chunk)
#             chunk=[i]
#     slices[r]=annot
# slices[40]
# print(slices)
