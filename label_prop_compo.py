#%%
from tkinter import Image
import sys
# sys.path.append('/home/nathan/SeqSeg/models/TorchIR')
from pytorch_lightning import Trainer
from soupsieve import select
from models.LabelProp_comp import LabelProp
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
    return torch.moveaxis(F.one_hot(Y), -1, dim).float()
size=(256,256)
dim=3
max_epochs=200
C=0.3
dataset='PLEX'
vis=''
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
losses={'compo-reg-up':True,'compo-reg-down':True,'compo-dice-up':True,'compo-dice-down':True,'bidir-cons-reg':False,'bidir-cons-dice':False}

ckpt_up=None#"/home/nathan/SeqSeg/voxelmorph_ckpts/labelprop/PLEX/up/labelprop-up-epoch=17-val_accuracy=0.52[0].ckpt"#"/home/nathan/SeqSeg/labelprop-epoch=96-val_accuracy=0.71-09022022-194748.ckpt"#'/home/nathan/SeqSeg/voxelmorph_ckpts/labelprop/PLEX/up/labelprop-up-epoch=59-val_accuracy=0.45[0].ckpt'#"/home/nathan/SeqSeg/voxelmorph_ckpts/labelprop/bench/bench/labelprop-epoch=180-val_accuracy=0.85-07022022-141236.ckpt"
#'/home/nathan/SeqSeg/voxelmorph_ckpts/labelprop/PLEX/up/labelprop-up-epoch=45-val_accuracy=0.22[0].ckpt'#"/home/nathan/SeqSeg/labelprop-epoch=96-val_accuracy=0.71-09022022-194748.ckpt"
resume_ckpt=None#'labelprop-up-epoch=145-val_accuracy=0.98[0, 1].ckpt'
ckpt_down="/home/nathan/SeqSeg/voxelmorph_ckpts/labelprop/bench/bench/labelprop-epoch=160-val_accuracy=0.96-08022022-143909.ckpt"#'labelprop-up-epoch=97-val_accuracy=0.84[1].ckpt'
selected_slices=[107,153,199]

if ckpt_up!=None:
    selected_slices=torch.load(ckpt_up)['hyper_parameters']['selected_slices']
    print('selected slices from ckpt',selected_slices)
start=np.min(selected_slices)
end=np.max(selected_slices)
#+list(range(80))[::10]#[107,160,199]#+list(range(101,185))[::2]#+[156]+[185]
data_PARAMS = {'batch_size':1,'subject_ids': [0], 'val_ids': [0], 'test_ids': [0],'aug':True,'dim':dim,'shape':size,'selected_slices':{'000':selected_slices}}
dm=DMDDataModule(**data_PARAMS,way='both')
logger=TensorBoardLogger("tb_logs", name="label_prop",log_graph=True)
model_up=LabelProp(n_classes=2,way='both',dim=dim,size=size,selected_slices=selected_slices,losses=losses)


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
    if resume_ckpt!=None:
        resume_ckpt=f'{dir}/up/{resume_ckpt}'
        model_up=model_up.load_from_checkpoint(resume_ckpt,strict=False)
    trainer=Trainer(gpus=1,max_epochs=max_epochs,logger=logger,callbacks=checkpoint_callback_up)
    trainer.fit(model_up,dm)
    model_up=model_up.load_from_checkpoint(checkpoint_callback_up.best_model_path)

else:
    ckpt_up=f'{ckpt_up}'
    model_up=model_up.load_from_checkpoint(ckpt_up,strict=False)
print(checkpoint_callback_up.best_model_path)

# dm=DMDDataModule(**data_PARAMS,way='up')


# data_PARAMS['dim']=3
dm.setup('fit')
_,Y_dense=dm.val_dataloader().dataset[0]
dm.setup('test')
X,Y=dm.test_dataloader().dataset[0]



chunks=[]
chunk=[]
if selected_slices!=None:
    for i in range(Y.shape[1]):
        if i not in selected_slices:
            Y[:,i,...]=Y[:,i,...]*0
        else:
            if len(torch.unique(torch.argmax(Y[:,i,...],1)))>1:
                chunk.append(i)
                if len(chunk)==2:
                    chunks.append(chunk)
                    chunk=[i]

print('CHUNKS',chunks)

Y_marks=torch.zeros_like(Y)
for i in range(Y.shape[1]):
    if len(torch.unique(torch.argmax(Y[:,i,...],0)))>1:
        Y_marks
        Y_marks[1,i,...]=torch.ones_like(Y_marks[1,i,...])
Y_sparse=new_nii(torch.argmax(Y,0).numpy()[start:end+1],'uint8')
ni.save(Y_sparse,'Y_sparse.nii.gz')
X=X[0]
X_out=deepcopy(X)
X_out2=deepcopy(X)

flows=torch.stack([X,X],1)
flows2=deepcopy(flows)
weights=torch.zeros((Y.shape[1]))
n=0
flag=False

for chunk in chunks:
    weights[chunk[0]:chunk[1]]=torch.FloatTensor(range(chunk[1]-chunk[0]))-(chunk[1]-chunk[0])/2

weights=(torch.arctan(C*weights)/3.14+0.5)
Y2=deepcopy(Y)
X_chunk=deepcopy(X)
X_chunk2=deepcopy(X)

# for i in range(Y.shape[1]):
#     y1=Y[:,i,...]
#     if len(torch.unique(torch.argmax(y1,0)))>1:
#         print(i)
model_up.to('cuda')
model_up.eval()
for i,x1 in enumerate(X):
    try:
        x2=X[i+1]
    except:
        print('End of volume')
    else:
        y1=Y[:,i,...]
        if len(torch.unique(torch.argmax(Y[:,i+1,...],0)))==1 and len(torch.unique(torch.argmax(y1,0)))>1:
            chunk_0=[x[0] for x in chunks if i in range(x[0],x[1])]
            chunk_1=[x[1] for x in chunks if i in range(x[0],x[1]+1)]
            x,y,trans=model_up.register_images(to_batch(x1,'cuda'),to_batch(x2,'cuda'),y1[None,...].to('cuda'))
            _,_,neg=model_up.register_images(to_batch(x2,'cuda'),to_batch(x1,'cuda'),y1[None,...].to('cuda'))
            X_out[i+1,...],Y[:,i+1,...]=x.cpu().detach()[0],y.cpu().detach()[0]#,trans.unsqueeze(0)
            if i in chunk_0:
                compo=[trans]
                compo_neg=[neg]
            else:
                compo.append(trans)
                compo_neg.append(neg)
                # flows[i+1:i+2,...]=model_up.compose_deformation(compo,trans)
            if i+1 in selected_slices or len(chunk_0)==0:
                X_chunk[i+1,...]=X[i+1,...]
            else:
                X_chunk[i+1,...]=model_up.registrator.transformer(to_batch(X[chunk_0[0]],'cuda'),model_up.compose_list(compo).to('cuda'))[0]
                Y[:,i+1,...]=model_up.registrator.transformer(Y[:,chunk_0[0],...][None,...].to('cuda'),model_up.compose_list(compo).to('cuda'))[0]


            # registrator=ImageRegistrator()
            # trans=registrator.register(to_batch(x1),to_batch(x2))
            # Y[i+1]=registrator.warp_src_into_dst(y1[None,...])[0].detach()
            #Y[i+1]=registrator.warp_src_into_dst(y1[None,...])[0]


for i in range(X.shape[0]-1,1,-1):
    x1=X[i]
    try:
        x2=X[i-1]
    except:
        print('End of volume')
    else:
        y1=Y2[:,i,...]
        
        if len(torch.unique(torch.argmax(y1,0)))>1 and len(torch.unique(torch.argmax(Y2[:,i-1,...],0)))==1:
           chunk_1=[x[1] for x in chunks if i in range(x[0],x[1]+1)]
           x,y,trans=model_up.register_images(to_batch(x1,'cuda'),to_batch(x2,'cuda'),y1[None,...].to('cuda'))
           X_out2[i-1,...],Y2[:,i-1,...]=x.cpu().detach()[0],y.cpu().detach()[0]
           print(chunk_1,[i])
           if i in chunk_1:
               negs=[trans]
           else:
               negs.append(trans)

           if i-1 in selected_slices or len(chunk_1)==0:
                X_chunk2[i-1,...]=X[i-1,...]
           else:
               
                X_chunk2[i-1,...]=model_up.registrator.transformer(to_batch(X[chunk_1[0]],'cuda'),model_up.compose_list(negs).to('cuda'))[0]
                Y2[:,i-1,...]=model_up.registrator.transformer(Y2[:,chunk_1[0],...][None,...].to('cuda'),model_up.compose_list(negs).to('cuda'))[0]

            #Y2[:,i-1,...]=model_up.registrator.transformer(y1[None,...].cuda(),flows[i:i+1].cuda())[0].cpu().detach()
#             registrator=ImageRegistrator()
#             trans=registrator.register(to_batch(x1),to_batch(x2))
#             Y2[i-1]=registrator.warp_src_into_dst(y1[None,...])[0].detach()

for i,w in enumerate(weights):
    Y[:,i,...]*=1-w
    Y2[:,i,...]*=w
    X_chunk[i]*=1-w
    X_chunk2[i]*=w
#%%
if selected_slices!=None:
    dices=[]
    dices2=[]
    dices_fused=[]
    mses_up=[]
    mses_down=[]
    for i in range(X.shape[0]):
        if i not in selected_slices and len(torch.unique(torch.argmax(Y_dense[:,i,...],0)))>1:
            dice=monai.metrics.compute_meandice(hardmax(Y[:,i,...],0)[None,...], Y_dense[:,i,...][None,...], include_background=False)
            dice2=monai.metrics.compute_meandice(hardmax(Y2[:,i,...],0)[None,...], Y_dense[:,i,...][None,...], include_background=False)
            dice_fused=monai.metrics.compute_meandice(hardmax(Y[:,i,...]+Y2[:,i,...],0)[None,...], Y_dense[:,i,...][None,...], include_background=False)

            dices.append(dice)
            dices2.append(dice2)
            dices_fused.append(dice_fused)
    #print(torch.nan_to_num(monai.metrics.compute_meandice(hardmax(Y,0)[:,100:186,...][None,...],Y_dense[:,100:186,...][None,...], include_background=False)).mean())
    print(torch.nan_to_num(torch.stack(dices)).mean())
    print(torch.nan_to_num(torch.stack(dices2)).mean())
    print(torch.nan_to_num(torch.stack(dices_fused)).mean())

    Y_dense=new_nii(torch.argmax(Y_dense,0)[start:end+1],'uint8')
    ni.save(Y_dense,'Y_dense.nii.gz')

#%%
import matplotlib.pyplot as plt   
weights=torch.zeros((Y_marks.shape[1]))
flag=False

# for i in range(Y_marks.shape[1]):

#     if Y_marks[1,i,0,0]==1:
#         if not flag: flag=True
#         else: 
#             weights[i-(n):i]=weights[i-(n):i]-n/2
#             weights[i]=i-n/2
#             n=1
#     else:
#         if flag:
#             weights[i]=n
#             n+=1


plt.plot(dices)
plt.plot(dices2)
# plt.plot(dices_fused)
plt.legend(['F','B'])
plt.figure()
plt.plot(dices_fused)
plt.figure()
plt.scatter(range(107,199),weights[107:199])
plt.scatter(range(107,199),1-weights[107:199])
#%%

disp=new_nii(X[start:end+1].detach().numpy()-X_out[start:end+1].detach().numpy())
X_chunk=new_nii(X_chunk[start:end+1].detach().numpy()+X_chunk2[start:end+1].detach().numpy())
X=new_nii(X[start:end+1].detach().numpy())
flows=new_nii(flows[start:end+1].detach().numpy())
X_out=new_nii((X_out[start:end+1].detach().numpy()+X_out2[start:end+1].detach().numpy())/2)
Y12=new_nii(torch.argmax(Y+Y2,0)[start:end+1].detach().numpy(),'uint8')
Y=new_nii(torch.argmax(Y,0)[start:end+1].detach().numpy(),'uint8')
Y2=new_nii(torch.argmax(Y2,0)[start:end+1].detach().numpy(),'uint8')
Y_marks=new_nii(torch.argmax(Y_marks,0)[start:end+1].detach().numpy(),'uint8')
ni.save(X,'X.nii.gz')
ni.save(flows,'flows_up.nii.gz')
ni.save(X_out,'X_out.nii.gz')
ni.save(Y,'Y_up.nii.gz')
ni.save(Y_marks,'Y_marks.nii.gz')
ni.save(Y2,'Y_down.nii.gz')
ni.save(Y12,'Y_weighted.nii.gz')
ni.save(disp,'disp.nii.gz')
ni.save(X_chunk,'X_chunk.nii.gz')