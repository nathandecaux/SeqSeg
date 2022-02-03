from tkinter import Image
import sys
from unittest import result
# sys.path.append('/home/nathan/SeqSeg/models/TorchIR')
from pytorch_lightning import Trainer

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

def is_already_done(file,res):
    if len(file)==0:
        return False
    for result in file:
        if result['losses']==res['losses'] and result['way']==res['way']:
            return True
    return False
        

size=(256,256)
dataset='PLEX'
r=2
ratio=str(r)
selected_slices=list(range(224))[::r]

try:
    with open('results.json') as f:
        results= json.load(f)
except:
    results=dict()

if dataset not in results.keys():
    results[dataset]=dict()
if ratio not in results[dataset].keys():
    print('noon')
    results[dataset][ratio]=[]

losses={'compo-reg-up':False,'compo-reg-down':False,'compo-dice-up':False,'compo-dice-down':False,'bidir-cons-reg':False,'bidir-cons-dice':False}

for compo_dice in [False,True]:
    res=dict()
    preds=dict()
    for way in ['up','down']:
        res=dict()
        losses[f'compo-dice-{way}']=compo_dice
        res['losses']=losses
        res['way']=way
        if is_already_done(results[dataset][ratio],res):
            print('skip')
        else:
            results[dataset][ratio].append(deepcopy(res))
            with open('results.json','w') as f:
                json.dump(results,f)
                

    res['losses']=losses
    res['way']='combined'
    if is_already_done(results[dataset][ratio],res):
        print('skip')
    else:
        results[dataset][ratio].append(deepcopy(res))
        with open('results.json','w') as f:
            json.dump(results,f)


# ckpt_up=None#'labelprop-up-epoch=150-val_accuracy=0.50[1].ckpt'#"labelprop-up-epoch=92-val_accuracy=0.92[1].ckpt"#'labelprop-up-epoch=139-val_accuracy=1.00[0, 1].ckpt'#'labelprop-up-epoch=145-val_accuracy=0.98[0, 1].ckpt'
# resume_ckpt=None#'labelprop-up-epoch=145-val_accuracy=0.98[0, 1].ckpt'
# ckpt_down='labelprop-up-epoch=150-val_accuracy=0.50[1].ckpt'#'labelprop-up-epoch=97-val_accuracy=0.84[1].ckpt'
# selected_slices=[107,160,199]#+list(range(101,185))[::2]#+[156]+[185]
# data_PARAMS = {'batch_size':1,'subject_ids': [0], 'val_ids': [0], 'test_ids': [0],'aug':True,'dim':dim,'shape':size,'lab':1,'selected_slices':{'000':selected_slices,'002':None}}
