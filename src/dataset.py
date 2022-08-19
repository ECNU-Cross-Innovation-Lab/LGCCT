import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

'''if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')'''
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        #self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        if split_type != 'test':
            self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        else:
            self.text = torch.tensor(dataset[split_type]['text'][:,:40,:].astype(np.float32)).cpu().detach()
        # self.text = torch.tensor(dataset[split_type]['text_bert'].astype(np.float32)).cpu().detach()
        print(self.text.shape)

        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        if split_type != 'test':
            self.audio = torch.tensor(self.audio).cpu().detach()
        else:
            self.audio = torch.tensor(self.audio[:,:40,:]).cpu().detach()
        print(self.audio.shape)

        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        # self.labels = torch.tensor(dataset[split_type]['classification_labels'].astype(np.float32)).cpu().detach()

        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.data = data

        self.n_modalities = 2 # text/ audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2]

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, (0, 0, 0)