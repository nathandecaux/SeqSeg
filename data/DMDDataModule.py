from configparser import Interpolation
import torch
import numpy as np
import os
from os.path import join
import numpy as np
import torch.utils.data as data
import scipy.io as sio
import nibabel as nb
import math
import torch.utils.data as tdata
from torch.utils.data.dataset import Dataset, TensorDataset
from torchio import sampler
import torchio
import kornia
from torchvision import transforms
from torchvision.transforms import functional as F
import pytorch_lightning as pl
import nibabel as ni
from torch.utils.data import DataLoader, ConcatDataset
import random
import torchio.transforms as tio
from inspect import getmembers, isfunction
from torch.nn import functional as func
from os import listdir
from os.path import join
import torch
from copy import deepcopy
import kornia.augmentation as K


class ConsistencyData(data.Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X.astype('float32')).unsqueeze(1)

    def __getitem__(self, index):
        x = self.X[index]
        x2 = x
        x2 = self.intensity(x2)
        affine_matrix = [-123]
        x2, affine_matrix = self.spatial(x2)

        return x, x2, affine_matrix

    def __len__(self):
        return len(self.X)

    def intensity(self, x):
        transform = tio.OneOf([tio.RandomMotion(translation=1), tio.RandomBlur(
        ), tio.RandomGamma(), tio.RandomSpike(intensity=[0.2, 0.5]), tio.RandomBiasField()])
        x = transform(x.unsqueeze(0)).squeeze(0)
        return x

    def spatial(self, x):
        affine_matrix = kornia.augmentation.RandomAffine(degrees=(-20, 20), translate=(
            0.2, 0.2), scale=(0.9, 1.1), keepdim=True).generate_parameters(x.unsqueeze(0).shape)

        affine_matrix = kornia.get_affine_matrix2d(**affine_matrix)
        x = kornia.geometry.warp_perspective(x.unsqueeze(
            0), affine_matrix, dsize=x.squeeze(0).shape).squeeze(0)
        return x, affine_matrix

class PlexDataVolume(data.Dataset):
    def __init__(self, X, Y,lab=1, mixup=0, aug=False):

        self.X = X.astype('float32')
        # Y= 1.*(Y==lab)
        Y[Y==11]=10
        self.Y=Y

        # idx_2_del=[]
        # for i in range(X.shape[0]):
        #     if len(np.unique(Y[i]))==1:
        #         idx_2_del.append(i)
        # for count,i in enumerate(idx_2_del):
        #     self.X=np.delete(self.X,i-count,0)
        #     self.Y=np.delete(self.Y,i-count,0)
        self.X = self.norm(torch.from_numpy(self.X))
        self.Y = torch.from_numpy(self.Y)
        
        self.aug = aug

        self.mixup = mixup

    def __getitem__(self, index):

        x = self.X[index]
        y = self.Y[index]
        x,y=self.resample(x,y,(300,300))
        # x = torch.from_numpy(x)
        # y = torch.from_numpy(y)
        # if self.aug != False :
        #     x, y = self.spatial(x, y)
        # x=self.norm(x)
        return x.unsqueeze(0), y
    
    def resample(self,X,Y,size):
        X=func.interpolate(X[None,None,...],size,mode='bilinear',align_corners=True)[0,0]
        Y=func.interpolate(Y[None,None,...],size,mode='nearest')[0,0]
        return X,Y
    def __len__(self):
        return len(self.Y)

    def norm(self, x):
        # x = (x-torch.mean(x))/torch.std(x)
        #x = (x-torch.mean(x))/torch.std(x)
        # return x
        norm = tio.RescaleIntensity((0, 1))
        if len(x.shape)==4:
            x = norm(x)
        elif len(x.shape)==3:
            x= norm(x[:, None, ...])[:,0, ...]
        else:
            x = norm(x[None, None, ...])[0, 0, ...]
        return x

class FullScan(data.Dataset):
    def __init__(self, X, Y,lab='all', mixup=0, aug=False,dim=3,way='up',shape=256,selected_slices=None):

        self.X = X.astype('float32')
        Y[Y==11]=10
        if isinstance(lab,int):
            Y= 1.*(Y==lab)
        self.Y=Y
        self.dim=dim
        # idx_2_del=[]
        # for i in range(X.shape[0]):
        #     if len(np.unique(Y[i]))==1:
        #         idx_2_del.append(i)
        # for count,i in enumerate(idx_2_del):
        #     self.X=np.delete(self.X,i-count,0)
        #     self.Y=np.delete(self.Y,i-count,0)
        self.X = self.norm(torch.from_numpy(self.X))[None,...]
        self.Y = torch.from_numpy(self.Y)[None,...]
        if isinstance(shape,int): shape=(shape,shape) 
        self.X,self.Y=self.resample(self.X,self.Y,(80,shape[0],shape[1]))
        if selected_slices!=None:
            for i in range(self.Y.shape[1]):
                if i not in selected_slices:
                    self.Y[:,i,...]=self.Y[:,i,...]*0
        self.Y=torch.moveaxis(func.one_hot(self.Y.long()), -1, 1).float()
        self.aug = aug
        self.way=way
        self.mixup = mixup

    def __getitem__(self, index):
 
        if self.dim==2:
            x = self.X[0,index]
            y = self.Y[0,:,index]
            idx=0
            if index==0:
                if self.way=='up':
                    idx=1
            elif index==self.Y.shape[2]-1:
                if self.way=='down':
                    idx=-1
            else:    
                if self.way=='up':
                    idx=1
                else:
                    idx=-1
            x2,y2=self.X[0,index+idx],self.Y[0,:,index+idx]
            # x,y=self.resample(x,y,(300,300))
            # x = torch.from_numpy(x)
            # y = torch.from_numpy(y)
            # if self.aug != False :
            #     x, y = self.spatial(x, y)
            # x=self.norm(x)
            return x.unsqueeze(0), y, x2.unsqueeze(0),y2
        else:
            x = self.X[index]
            y = self.Y[index]
            return x.unsqueeze(0), y
    def resample(self,X,Y,size):
        X=func.interpolate(X[None,...],size,mode='trilinear',align_corners=True)[0]
        Y=func.interpolate(Y[None,...],size,mode='nearest')[0]
        return X,Y
    def __len__(self):
        if self.dim==3:
            return len(self.Y)
        if self.dim==2:
            return self.Y.shape[2]
    def norm(self, x):
        # x = (x-torch.mean(x))/torch.std(x)
        #x = (x-torch.mean(x))/torch.std(x)
        # return x
        norm = tio.RescaleIntensity((0, 1))
        if len(x.shape)==4:
            x = norm(x)
        elif len(x.shape)==3:
            x= norm(x[:, None, ...])[:,0, ...]
        else:
            x = norm(x[None, None, ...])[0, 0, ...]
        return x

class RegisterDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
 

    def __getitem__(self, index):
        x,y = self.dataset[0]
        idx=1
        #while idx==index:
        #    idx=int(torch.randint(low=0,high=self.__len__(),size=(1,))[0])
        x2,y2=self.dataset[idx]
        return x, y, x2, y2


    def __len__(self):
        return 1#len(self.dataset)

    def norm(self, x):
        # x = (x-torch.mean(x))/torch.std(x)
        #x = (x-torch.mean(x))/torch.std(x)
        # return x
        norm = tio.RescaleIntensity((0, 1))
        if len(x.shape)==4:
            x = norm(x)
        elif len(x.shape)==3:
            x= norm(x[:, None, ...])[:,0, ...]
        else:
            x = norm(x[None, None, ...])[0, 0, ...]
        return x

class PlexData(data.Dataset):
    def __init__(self, X, Y,lab=1, mixup=0, aug=False):

        self.X = X.astype('float32')
        Y[Y==11]=10
        self.Y=Y
        print(np.unique(Y))
        idx_2_del=[]
        for i in range(X.shape[0]):
            if len(np.unique(Y[i]))==1:
                idx_2_del.append(i)
        for count,i in enumerate(idx_2_del):
            self.X=np.delete(self.X,i-count,0)
            self.Y=np.delete(self.Y,i-count,0)
        self.X = self.norm(torch.from_numpy(self.X))
        self.Y = torch.from_numpy(self.Y)
        self.aug = aug

        self.mixup = mixup
        

    def __getitem__(self, index):

        x = self.X[index]
        y = self.Y[index]
        x,y=self.resample(x,y,(300,300))

        # x = torch.from_numpy(x)
        # y = torch.from_numpy(y)
        if self.aug != False :
            x, y = self.spatial(x, y)
        x=self.norm(x)
        return x.unsqueeze(0), y

    def __len__(self):
        return len(self.Y)

    def resample(self,X,Y,size):
        X=func.interpolate(X[None,None,...],size,mode='bilinear',align_corners=True)[0,0]
        Y=func.interpolate(Y[None,None,...],size,mode='nearest')[0,0]
        return X,Y

    def norm(self, x):
        # x = (x-torch.mean(x))/torch.std(x)
        #x = (x-torch.mean(x))/torch.std(x)
        # return x
        norm = tio.RescaleIntensity((0, 1))
        if len(x.shape)==4:
            x = norm(x)
        elif len(x.shape)==3:
            x= norm(x[:, None, ...])[:,0, ...]
        else:
            x = norm(x[None, None, ...])[0, 0, ...]
        return x

    def spatial(self,x,y):
        trans = K.AugmentationSequential(K.RandomAffine(degrees=[-20,20], scale=[0.8,1.2],shear=[-20,20], resample="nearest", p=0.9), data_keys=["input", "mask"])
        x,y=trans(x[None,None,:,:],y[None,None,:,:])
        return x[0,0,...],y[0,0,...]      



class SemiPlexData(data.Dataset):
    def __init__(self, X,size=None):

        # self.X = X[:size].unsqueeze(1)
        self.X=X.unsqueeze(1)
        self.size=size
        # trans=K.AugmentationSequential(K.RandomSolarize(p=1),K.RandomBoxBlur(p=1),data_keys=["input"])
        # self.X2= trans(self.X).squeeze(1)
    def __getitem__(self, index):
        # index=index+np.random.randint(0,1000)
        x = self.X[index]
        x=self.norm(x)
        x2 = 254*torch.ones_like(x).squeeze(0)
        return x, x2

    def __len__(self):
        return self.size#len(self.X)
    
    def norm(self, x):
        norm = tio.RescaleIntensity((0, 1))
        x = norm(x[None, ...])[0,...]
        return x



class DMDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/mnt/Data/PLEX/datasets/bids/', supervised=True, subject_ids='all',val_ids=[1],test_ids=[1], limb='both', batch_size=8, mixup=0, aug=False,interpolate=False, disentangled=False,dim=2,way='up',register=False,shape=256,selected_slices=None):
        super().__init__()
        self.data_dir = '/home/nathan/Datasets/DMD/'
        if subject_ids == 'all':
            subject_ids = range(2, 12)
        indices = []
        # if not isinstance(subject_ids, list):
        #     subject_ids=[int(subject_ids)]
        self.disentangled=disentangled
        self.subject_ids=subject_ids
        self.val_ids=val_ids
        self.test_ids=test_ids
        self.limb = limb
        self.supervised = supervised
        self.batch_size = batch_size
        self.mixup = float(mixup)
        self.aug = bool(aug)
        self.transforms = None
        self.interpolate = interpolate
        self.sampler=None
        self.dim=dim
        self.shape=shape
        self.way=way        
        self.selected_slices=selected_slices

        self.register=register
    def setup(self, stage=None):
        plex_train=[]
        plex_val=[]
        plex_test=[]
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None or stage == 'get' and not self.deepgrow :

            for idx in self.subject_ids:
                idx=str(idx)
                idx2=str(idx).zfill(2)
                data_train = (
                    ni.load(join(self.data_dir, f'sub-{idx2}/ses-01/anat/sub-{idx}_ses-1_DIXON6ECHOS-e3.nii.gz'))).get_fdata()
                
                mask_train = (
                    ni.load(join(self.data_dir, f'sub-{idx2}/ses-01/anat/seg.nii.gz'))).get_fdata()
                # data_train=data_train[160:320,90:220,:]
                # print(mask_train.shape)
                # mask_train=mask_train[160:320,90:220,:]
                if idx=='0':
                    data_train=data_train[550:900,0:290,190:475]
                    mask_train=mask_train[550:900,0:290,190:475]
                data_train=np.moveaxis(data_train,-1,0)
                mask_train=np.moveaxis(mask_train,-1,0)
                if False:#self.dim==2:
                    plex_train.append(PlexData(
                        data_train, mask_train, mixup=self.mixup, aug=self.aug))  # self.aug)
                else: 
                    if idx not in self.selected_slices: selected_slices=None
                    else: selected_slices=self.selected_slices[idx]
                    print('waohou',selected_slices)
                    plex_train.append(FullScan(
                        data_train, mask_train, mixup=self.mixup, aug=self.aug,dim=self.dim,way=self.way,shape=self.shape,selected_slices=selected_slices))  # self.aug)
            for idx in self.val_ids:
                idx=str(idx)
                idx2=str(idx).zfill(2)
                
                data_train = (
                    ni.load(join(self.data_dir, f'sub-{idx2}/ses-01/anat/sub-{idx}_ses-1_DIXON6ECHOS-e3.nii.gz'))).get_fdata()
                
                mask_train = (
                    ni.load(join(self.data_dir, f'sub-{idx2}/ses-01/anat/seg.nii.gz'))).get_fdata()
                mask_train[mask_train==11]=10
                # data_train=data_train[160:320,90:220,:]
                # print(mask_train.shape)
                # mask_train=mask_train[160:320,90:220,:]
                data_train=np.moveaxis(data_train,-1,0)
                mask_train=np.moveaxis(mask_train,-1,0)
                if False:#self.dim==2:
                    plex_val.append(PlexData(
                        data_train, mask_train))  # self.aug)
                else:
                    plex_val.append(FullScan(
                        data_train, mask_train,dim=3,way=self.way,shape=self.shape))  # self.aug)



            self.plex_train = ConcatDataset(plex_train)

            self.plex_val = ConcatDataset(plex_val)

            if self.register:
                self.plex_train=RegisterDataset(self.plex_train)
                self.plex_val=RegisterDataset(self.plex_val)
            print(f'Dataset Size : {len(self.plex_train)}')
            print(f'Val Size : {len(self.plex_val)}')

        # Assign Test split(s) for use in Dataloaders
        if stage == 'test':
            for idx in self.test_ids:
                idx=str(idx)
                idx2=str(idx).zfill(2)
                data_train = (
                    ni.load(join(self.data_dir, f'sub-{idx2}/ses-01/anat/sub-{idx}_ses-1_DIXON6ECHOS-e3.nii.gz'))).get_fdata()
                
                mask_train = (
                    ni.load(join(self.data_dir, f'sub-{idx2}/ses-01/anat/seg.nii.gz'))).get_fdata()
                # data_train=data_train[160:320,90:220,:]
                # print(mask_train.shape)
                # mask_train=mask_train[160:320,90:220,:]
                data_train=np.moveaxis(data_train,-1,0)
                mask_train=np.moveaxis(mask_train,-1,0)
                mask_train[mask_train==11]=10

                if self.dim==2:
                    plex_test.append(PlexDataVolume(
                        data_train, mask_train))  # self.aug)
                else:
                    plex_test.append(FullScan(
                        data_train, mask_train,shape=self.shape))  # self.aug)

            # print(len(plex_test[type]))

            self.plex_test=ConcatDataset(plex_test)


            self.dims = self.plex_test[0][0].shape
    
    
    def train_dataloader(self,batch_size=None):
        if batch_size==None: batch_size=self.batch_size
        return DataLoader(self.plex_train, batch_size, num_workers=8, shuffle=False,pin_memory=False,sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.plex_val, 1, num_workers=8, pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(self.plex_test, 1,shuffle=False, num_workers=8, pin_memory=False)
