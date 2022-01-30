from atexit import register
from tkinter import Image
import sys
sys.path.append('/home/nathan/DeepSeg/models/TorchIR')
from pytorch_lightning import Trainer
from data.DMDDataModule import DMDDataModule
from kornia.geometry import ImageRegistrator
from models.ScanRegister import ScanRegister
import torch
import numpy as np
import nibabel as ni
import kornia.geometry as KG
import torch.nn.functional as F
from copy import deepcopy
from voxelmorph.torch.networks import VxmDense
from pytorch_lightning.loggers import TensorBoardLogger

def new_nii(X,type='float32'):
    return ni.Nifti1Image(np.array(X).astype(type),np.eye(4))
    
def to_batch(x,device='cpu'):
    return x[None,None,...].to(device)

def hardmax(Y,dim):
    return torch.moveaxis(F.one_hot(torch.argmax(Y,dim), 11), -1, dim).float()
def to_one_hot(Y,dim=1):
    return torch.moveaxis(F.one_hot(Y, 11), -1, dim).float()
dim=3
max_epochs=100

shape=128
# for i in range(13,21):
#     if i!=17:
data_PARAMS = {'batch_size':1,'subject_ids': [17,19], 'val_ids': [17,19], 'test_ids': [17,19],'aug':True,'dim':dim,'shape':shape}
dm=DMDDataModule(**data_PARAMS,way='up',register=True)
logger=TensorBoardLogger("tb_logs", name="label_prop",log_graph=True)
model_up=ScanRegister(way='up',dim=dim,shape=shape,learning_rate=1e-3)
trainer=Trainer(gpus=1,max_epochs=max_epochs,logger=logger)
trainer.fit(model_up,dm)
dm.setup('test')
X,Y=dm.test_dataloader().dataset[0]
X2,_=dm.test_dataloader().dataset[1]
X,Y,X2=X.unsqueeze(0),Y.unsqueeze(0),X2.unsqueeze(0)
X_out,Y_hat,trans=model_up.register_images(X,X2,Y)
X_out,Y_hat,trans=X_out.cpu().detach()[0],Y_hat.cpu().detach()[0],trans.cpu().detach()[0]

dir='registered/'
X2=new_nii(X2[0,0].detach().numpy())
X_out=new_nii((X_out[0].detach().numpy()))
Y_hat=new_nii(torch.argmax(Y_hat,0).detach().numpy(),'uint8')

ni.save(X2,f'{dir}X2.nii.gz')
ni.save(X_out,f'{dir}X_out.nii.gz')
ni.save(Y_hat,f'{dir}Y_hat.nii.gz')

