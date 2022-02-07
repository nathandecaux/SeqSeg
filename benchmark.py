from tkinter import Image
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
    with open('results.json','w') as f:
        json.dump(results,f, cls=NumpyArrayEncoder)


size=(256,256)

dataset='PLEX'

data_PARAMS = {'dataset':dataset,'batch_size':1,'subject_ids': [0], 'val_ids': [0], 'test_ids': [0],'aug':True,'dim':3,'shape':size,'lab':1}
model_PARAMS= {'size':size}



try:
    with open('results.json') as f:
        results= json.load(f)
except:
    results=dict()

if dataset not in results.keys():
    results[dataset]=dict()

for r in [2,3,5]:
    ratio=str(r)
    selected_slices=list(range(224))[::r]
    if ratio not in results[dataset].keys():
        print('noon')
        results[dataset][ratio]=[]
    losses={'compo-reg-up':False,'compo-reg-down':False,'compo-dice-up':False,'compo-dice-down':False,'bidir-cons-reg':True,'bidir-cons-dice':True}

    data_PARAMS['selected_slices']={'000':selected_slices}
    for compo_reg in [True]:
        for compo_dice in [True]:
            res=dict()
            preds=dict()
            for way in ['both']:
                res=dict()
                if way=='both':
                    for k in ['up','down']:
                        losses[f'compo-dice-{k}']=True
                        losses[f'compo-reg-{k}']=True
                else:
                    losses[f'compo-dice-{way}']=compo_dice
                    losses[f'compo-reg-{way}']=compo_reg
                res['losses']=losses
                res['way']=way
                model_PARAMS.update({'losses':losses,'way':way,'selected_slices':selected_slices})
                if is_already_done(results[dataset][ratio],res):
                    print('skip')
                    Y_up,Y_down,res_train=train_and_eval(data_PARAMS,model_PARAMS,ckpt=[x['ckpt'] for x in results[dataset][ratio] if x['losses']==losses and x['way']==way][0])
                    if way=='up': preds[way]={'Y':Y_up,'ckpt':res_train['ckpt']}
                    else: preds[way]={'Y':Y_down,'ckpt':res_train['ckpt']}
                else:    
                    Y_up,Y_down,res_train=train_and_eval(data_PARAMS,model_PARAMS)
                    if way=='up': preds[way]={'Y':Y_up,'ckpt':res_train['ckpt']}
                    else: preds[way]={'Y':Y_down,'ckpt':res_train['ckpt']}
                    res.update(res_train)
                    results[dataset][ratio].append(deepcopy(res))
                    save_json(results,f)
                        
            # res_train=get_dices(get_test_data(data_PARAMS)[1],preds['up']['Y'],preds['down']['Y'],selected_slices)
            # res['losses']=losses
            # res['way']='combined'
            # res.update(res_train)
            # res['ckpt']=preds['up']['ckpt']+' + '+preds['down']['ckpt']
            # if is_already_done(results[dataset][ratio],res):
            #     print('skip')
            # else:
            #     results[dataset][ratio].append(deepcopy(res))
            #     save_json(results,f)


# ckpt_up=None#'labelprop-up-epoch=150-val_accuracy=0.50[1].ckpt'#"labelprop-up-epoch=92-val_accuracy=0.92[1].ckpt"#'labelprop-up-epoch=139-val_accuracy=1.00[0, 1].ckpt'#'labelprop-up-epoch=145-val_accuracy=0.98[0, 1].ckpt'
# resume_ckpt=None#'labelprop-up-epoch=145-val_accuracy=0.98[0, 1].ckpt'
# ckpt_down='labelprop-up-epoch=150-val_accuracy=0.50[1].ckpt'#'labelprop-up-epoch=97-val_accuracy=0.84[1].ckpt'
# selected_slices=[107,160,199]#+list(range(101,185))[::2]#+[156]+[185]
# data_PARAMS = {'batch_size':1,'subject_ids': [0], 'val_ids': [0], 'test_ids': [0],'aug':True,'dim':dim,'shape':size,'lab':1,'selected_slices':{'000':selected_slices,'002':None}}
