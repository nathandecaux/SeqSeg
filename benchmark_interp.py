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


size=(288,288)

dataset='DMD'

data_PARAMS = {'dataset':dataset,'batch_size':1,'subject_ids': [17], 'val_ids': [17], 'test_ids': [17],'aug':True,'dim':3,'shape':size}
model_PARAMS= {'model_name':'Interp'}



try:
    with open('results_interp.json') as f:
        results= json.load(f)
except:
    results=dict()

if dataset not in results.keys():
    results[dataset]=dict()
slices=dict()
for r in [46]:
    if str(r) not in results[dataset].keys():
        ratio=str(r)
        selected_slices=[21,67]
        # if r%2==0:
        #     selected_slices+=list(range(80))[1::r]
        # else:
        #     selected_slices+=list(range(80))[::r]
        slices[r]=selected_slices


        data_PARAMS['selected_slices']={'17':selected_slices}
        res=dict()
        preds=dict()
        Y_up,Y_down,res_train=train_and_eval(data_PARAMS,model_PARAMS)
        res.update(res_train)
        results[dataset][ratio]=deepcopy(res)
        save_json(results,f)



