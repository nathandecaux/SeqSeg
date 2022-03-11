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
    with open('results_interp.json','w') as f:
        json.dump(results,f, cls=NumpyArrayEncoder)
def new_nii(X,type='float32'):
    affine=np.eye(4)
    if dataset=='PLEX':
        affine[0,0]=5/3
    else:
        affine[0,0]=5
    return ni.Nifti1Image(np.array(X).astype(type),affine)

size=(256,256)

dataset='PLEX'

data_PARAMS = {'dataset':dataset,'batch_size':1,'subject_ids': [0], 'val_ids': [0], 'test_ids': [0],'aug':True,'dim':3,'shape':size}
model_PARAMS= {'model_name':'Interp','mode':'morph'}



with open('results.json') as f:
    res_prop= json.load(f)


try:
    with open('results_interp.json') as f:
        results= json.load(f)
except:
    results=dict()

if dataset not in results.keys():
    results[dataset]=dict()

losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}

slices=dict()

# for r in res_prop[dataset].keys():
#     print(r)
#     for exp in res_prop[dataset][r]:
#         if exp['losses']==losses:
#             print(exp['way'])
#             if exp['way']=='both':
#                 selected_slices=torch.load(exp['ckpt'])['hyper_parameters']['selected_slices']
#     print(selected_slices)
#     slices[r]=len([x for x in selected_slices if x>106 and x<200])

    # print(torch.load(results[dataset][r]['ckpt'])['hyper_parameters'])
for r in [46]:
    if True:#str(r) not in results[dataset].keys():
        ratio=str(r)
        # selected_slices=[21,67]
        # if r%2==0:
        #     selected_slices+=list(range(80))[1::r]
        # else:
        #     selected_slices+=list(range(80))[::r]
        selected_slices=[107, 153, 199]#[107,153,199]
        slices[r]=selected_slices

        model_PARAMS['selected_slices']=selected_slices
        data_PARAMS['selected_slices']={'000':selected_slices}


        res=dict()
        preds=dict()
        Y_up,Y_down,res_train=train_and_eval(data_PARAMS,model_PARAMS)
        Y_fused=torch.argmax(Y_up,0)
        Y_fused=new_nii(Y_fused.cpu().detach().numpy()[107:200],'uint8')
        ni.save(Y_fused,'Y_Interp.nii.gz')
    #     res.update(res_train)
    #     results[dataset][ratio]=deepcopy(res)
    #     save_json(results,f)

print(slices)
# for r in [2,3,5,10,20,46]:
#     chunks=[]
#     chunk=[]
#     annot=0
#     for i in range(107,200):
#         if i in slices[r]:
#             chunk.append(i)
#             annot+=1
#         if len(chunk)==2:
#             chunks.append(chunk)
#             chunk=[i]
#     slices[r]=annot
# print(slices.values())
