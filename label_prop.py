from tkinter import Image
import sys
# sys.path.append('/home/nathan/SeqSeg/models/TorchIR')
from pytorch_lightning import Trainer

from kornia.geometry import ImageRegistrator
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

    
def to_batch(x,device='cpu'):
    return x[None,None,...].to(device)

def hardmax(Y,dim):
    return torch.moveaxis(F.one_hot(torch.argmax(Y,dim)), -1, dim)
def to_one_hot(Y,dim=1):
    return torch.moveaxis(F.one_hot(Y, 11), -1, dim).float()
size=(304,304)
dim=3
max_epochs=200
C=1
dataset='PLEX'

def new_nii(X,type='float32'):
    affine=np.eye(4)
    if dataset=='PLEX':
        affine[0,0]=5/3
    else:
        affine[0,0]=5
    return ni.Nifti1Image(np.array(X).astype(type),affine)

if dataset=='PLEX':
    from data.CleanDataModule import PlexDataModule as DMDDataModule
else:
    from data.DMDDataModule import DMDDataModule


dir=f'/home/nathan/SeqSeg/voxelmorph_ckpts/labelprop/{dataset}'
ckpt_up=None#'labelprop-up-epoch=86-val_accuracy=0.30[0].ckpt'#'labelprop-up-epoch=98-val_accuracy=0.85[1].ckpt'#'labelprop-up-epoch=99-val_accuracy=0.58[19].ckpt'
ckpt_down='labelprop-down-epoch=00-val_accuracy=0.11[1].ckpt'#'labelprop-up-epoch=97-val_accuracy=0.84[1].ckpt'
selected_slices=[100]+[128]#+list(range(101,185))[::2]#+[156]+[185]
data_PARAMS = {'batch_size':1,'subject_ids': [0], 'val_ids': [0], 'test_ids': [0],'aug':True,'dim':dim,'shape':size,'lab':1,'selected_slices':selected_slices}
dm=DMDDataModule(**data_PARAMS,way='up')
logger=TensorBoardLogger("tb_logs", name="label_prop",log_graph=True)
model_up=LabelProp(way='both',dim=dim,size=size,selected_slices=selected_slices)
model_down=LabelProp(way='down',dim=dim,size=size)

checkpoint_callback_up = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath=f'{dir}/up',
        filename='labelprop-up-{epoch:02d}-{val_accuracy:.2f}'+str(data_PARAMS['subject_ids']),
        save_top_k=1,
        mode='max',
    )

checkpoint_callback_down = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath=f'{dir}/down',
        filename='labelprop-down-{epoch:02d}-{val_accuracy:.2f}'+str(data_PARAMS['subject_ids']),
        save_top_k=1,
        mode='max',
    )

if ckpt_up==None:
    trainer=Trainer(gpus=1,max_epochs=max_epochs,logger=logger,callbacks=checkpoint_callback_up)
    trainer.fit(model_up,dm)
    model_up=model_up.load_from_checkpoint(checkpoint_callback_up.best_model_path)

else:
    ckpt_up=f'{dir}/up/{ckpt_up}'
    model_up=model_up.load_from_checkpoint(ckpt_up,strict=False)
print(checkpoint_callback_up.best_model_path)

dm=DMDDataModule(**data_PARAMS,way='up')

if ckpt_down==None:
    trainer=Trainer(gpus=1,max_epochs=max_epochs,callbacks=checkpoint_callback_down)
    trainer.fit(model_down,dm)
    model_down=model_down.load_from_checkpoint(checkpoint_callback_down.best_model_path)
else:
    ckpt_down=f'{dir}/up/{ckpt_down}'
    model_down=model_down.load_from_checkpoint(ckpt_down,strict=False)

data_PARAMS['dim']=3
dm.setup('fit')
_,Y_dense=dm.val_dataloader().dataset[0]
dm.setup('test')
X,Y=dm.test_dataloader().dataset[0]
if selected_slices!=None:
    for i in range(Y.shape[1]):
        if i not in selected_slices:
            Y[:,i,...]=Y[:,i,...]*0
Y_sparse=new_nii(torch.argmax(Y,0).numpy(),'uint8')
ni.save(Y_sparse,'Y_sparse.nii.gz')
X=X[0]
X_out=deepcopy(X)
X_out2=deepcopy(X)

flows=torch.stack([X,X],-1)
flows2=deepcopy(flows)
weights=torch.zeros((Y.shape[1]))
n=0
flag=False
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

weights=(torch.arctan(C*weights)/3.14+0.5)
Y2=deepcopy(Y)

for i in range(Y.shape[1]):
    y1=Y[:,i,...]
    if len(torch.unique(torch.argmax(y1,0)))>1:
        print(i)
model_up.to('cuda')

for i,x1 in enumerate(X):
    try:
        x2=X[i+1]
    except:
        print('End of volume')
    else:
        y1=Y[:,i,...]
        if len(torch.unique(torch.argmax(Y[:,i+1,...],0)))==1 and len(torch.unique(torch.argmax(y1,0)))>1:
            x,y,trans=model_up.register_images(to_batch(x1,'cuda'),to_batch(x2,'cuda'),y1[None,...].to('cuda'))
            X_out[i+1,...],Y[:,i+1,...],flows[i+1,...]=x.cpu().detach()[0],y.cpu().detach()[0],torch.moveaxis(trans.cpu().detach()[0],0,-1)
            # registrator=ImageRegistrator()
            # trans=registrator.register(to_batch(x1),to_batch(x2))
            # Y[i+1]=registrator.warp_src_into_dst(y1[None,...])[0].detach()
            #Y[i+1]=registrator.warp_src_into_dst(y1[None,...])[0]

model_down.to('cuda')
for i in range(X.shape[0]-1,1,-1):
    x1=X[i]
    try:
        x2=X[i-1]
    except:
        print('End of volume')
    else:
        y1=Y2[:,i,...]
        if len(torch.unique(torch.argmax(y1,0)))>1 and len(torch.unique(torch.argmax(Y2[:,i-1,...],0)))==1:
            x,y,trans=model_down.register_images(to_batch(x1,'cuda'),to_batch(x2,'cuda'),y1[None,...].to('cuda'))
            X_out2[i-1,...],Y2[:,i-1,...],flows2[i-1,...]=x.cpu().detach()[0],y.cpu().detach()[0],torch.moveaxis(trans.cpu().detach()[0],0,-1)

#             registrator=ImageRegistrator()
#             trans=registrator.register(to_batch(x1),to_batch(x2))
#             Y2[i-1]=registrator.warp_src_into_dst(y1[None,...])[0].detach()

for i,w in enumerate(weights):
    Y[:,i,...]*=1-w
    Y2[:,i,...]*=w


if selected_slices!=None:
    dices=[]
    for i in range(X.shape[0]):
        if i not in selected_slices and len(torch.unique(torch.argmax(Y_dense[:,i,...],0)))>1:
            dice=monai.metrics.compute_meandice(hardmax(Y[:,i,...]+Y2[:,i,...],0)[None,...], Y_dense[:,i,...][None,...], include_background=False)
            dices.append(dice)
    print(torch.nan_to_num(torch.stack(dices)).mean())
    Y_dense=new_nii(torch.argmax(Y_dense,0),'uint8')
    ni.save(Y_dense,'Y_dense.nii.gz')



X=new_nii(X.detach().numpy())
# flows=new_nii(flows.detach().numpy())
X_out=new_nii((X_out.detach().numpy()+X_out2.detach().numpy())/2)
Y12=new_nii(torch.argmax(Y+Y2,0).detach().numpy(),'uint8')
Y=new_nii(torch.argmax(Y,0).detach().numpy(),'uint8')
Y2=new_nii(torch.argmax(Y2,0).detach().numpy(),'uint8')
ni.save(X,'X.nii.gz')
# ni.save(flows,'flows_up.nii.gz')
ni.save(X_out,'X_out.nii.gz')
ni.save(Y,'Y_up.nii.gz')
ni.save(Y2,'Y_down.nii.gz')
ni.save(Y12,'Y_weighted.nii.gz')