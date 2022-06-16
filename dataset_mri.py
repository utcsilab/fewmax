from dataclasses import replace
from logging import root
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import random
import torch

class MRI_Patch(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir='/csiNAS/ali/temp/brain_patch_200k_90_90_10_90/patch_data/', augment_mag=True, augment_ang=True, decoder=False, data_type='brain', ratio=None, training=True):
        self.root_dir = root_dir
        self.data_type = data_type
        self.training = training
        self.ratio = ratio

        """
        #subjs = [i for i in range(401, 451)]
        #random_index = np.random.choice(400, 10, replace=False)

        subjs = [46, 120, 187, 165, 95, 4, 68, 47, 106, 166, 54, 381, 60, 295, 323, 161, 145, 12, 236, 176]
        slcs = [0, 1, 2, 3, 4]
        self.data_name_list = []
        for i in subjs:
            for j in slcs:
            #random_index = np.random.choice(10, 5, replace=False)
                for k in range(10):
                    a = [x for x in glob.glob(os.path.join(root_dir, f'{i}_{j}_{k}.npy'))]
                    self.data_name_list.append(a)
        a = np.asarray(self.data_name_list).reshape(-1, 1)
        np.save('nasty_dataset/brain_fewshot_data_dir', a)
        exit()


        #random_index = np.random.choice(len(self.data_name), 2000, replace=True)
        indicies = np.load('nasty_dataset/brain_fewshot_data_dir.npy')
        self.data_list = []
        for ind in indicies:
            print(ind)
            data_i = np.load(ind[0])
            self.data_list.append([data_i.real, data_i.imag])
        self.data = np.asarray(self.data_list)
        np.save('nasty_dataset/dataset_brain_fewshot.npy', self.data)
        print(self.data.shape)
        exit()
        """


        #print(self.data.shape)
        #mean = np.mean(self.data, axis=(0, 2, 3))
        #std = np.std(self.data, axis=(0, 2, 3))
        #exit()

        #self.data[:, 0] = (self.data[:, 0] - mean[0])/std[0]
        #self.data[:, 1] = (self.data[:, 1] - mean[1])/std[1]

        if self.data_type == 'brain':
            if self.training:
                if self.ratio == 0.05:
                    self.data = np.load('data_mri/dataset_brain_fewshot.npy')
                else:
                    self.data = np.load('data_mri/dataset_brain_train.npy')
            else:
                self.data = np.load('data_mri/dataset_brain_test.npy')


        elif self.data_type == 'knee':
            if self.training:
                if self.ratio == 0.05:
                    self.data = np.load('data_mri/dataset_knee_fewshot.npy')
                else:
                    self.data = np.load('data_mri/dataset_knee_train.npy')
            else:
                self.data = np.load('data_mri/dataset_knee_test.npy')
        else:


            raise 'Check the name of dataset again'
        
        self.data_complex = self.data[:, 0] + 1j*self.data[:, 1]
        self.data_abs = np.abs(self.data_complex)
        self.max = np.max(self.data_abs, axis=(1, 2)).reshape(len(self.data_abs), 1, 1)
        self.data_complex = self.data_complex/(self.max + 1e-4)
        self.augment_mag = augment_mag
        self.agument_ang = augment_ang 
        self.decoder = decoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data_complex[index]

        if self.decoder:
            return torch.tensor(np.asarray([img.real, img.imag]))

        img_1, img_2  = img, img

        if self.augment_mag:
            img_1 = self.augt_mag(img_1)
            img_2 = self.augt_mag(img_2)
        if self.agument_ang:
            img_1 = self.augt_ang(img_1)
            img_2 = self.augt_ang(img_2)
        img_1 = torch.tensor(np.asarray([img_1.real, img_1.imag]))
        img_2 = torch.tensor(np.asarray([img_2.real, img_2.imag]))

        return index, [img_1, img_2]

    def augt_ang(self, image):
        c = random.uniform(0, 1)
        c = np.exp(1j*2*np.pi*c)
        return image*c


    def augt_mag(self, image):
        c = random.uniform(0.9, 1.1)
        return image*c

if __name__ == '__main__':
    dataset_ = MRI_Patch(data_type='brain',decoder=True, training=False)
    print(dataset_.__len__())
    exit()
    f, axarr = plt.subplots(5, 20)
    random_index = np.random.choice(200, 20, replace=False)

    while True:
        cnt = 0
        for j in range(100):
            images_ = dataset_.__getitem__(400+j)
            anchor = torch.sqrt(images_[0].pow(2) + images_[1].pow(2)).detach().cpu().numpy()
            if j%20 == 0 and j!=0:
                cnt +=1
            axarr[cnt, j%20].imshow(anchor, cmap='gray')
            axarr[cnt, j%20].xaxis.set_visible(False)
            axarr[cnt, j%20].yaxis.set_visible(False)
            f.suptitle('brain test whole subject (105)', fontname='serif')
        #plt.savefig('figures_nn_knee_trained_on_1000_knee_test/fig_' +
        #            str(batch_idx)+'_'+'.png')
        plt.savefig('brain_test_com_sub_4.pdf', dpi=1000)
        plt.close()
        exit()
