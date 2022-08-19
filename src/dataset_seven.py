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
        self.len = 0
        if data == "seven":
            self.data_path = os.path.join("E:\data-processed-icassp-20.tar\processed\IEMOCAP\seven_category_120")
            self.dataset = self.load_data(split_type)
            self.len = len(self.dataset)
        else:
            dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
            dataset = pickle.load(open(dataset_path, 'rb'))


            # These are torch tensors
            #self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
            self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
            self.audio = dataset[split_type]['audio'].astype(np.float32)
            self.audio[self.audio == -np.inf] = 0
            self.audio = torch.tensor(self.audio).cpu().detach()
            self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

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
        if self.data == "seven":

            N_CATEGORY = 7
            encoder_size_audio = 74
            encoder_size_text = 300

            mfcc, seqN_audio, prosody, trans, label = self.dataset[index]

            audio_mix = mfcc
            encoder_inputs_audio = audio_mix[:encoder_size_audio]
            encoder_seq_audio = np.min((seqN_audio, encoder_size_audio))
            encoder_prosody = prosody

            # Text
            seqN_text = 0
            tmp_index = np.where(trans == 0)[0]  # find the pad index

            if (len(tmp_index) > 0):  # pad exists
                seqN_text = np.min(tmp_index[0], encoder_size_text)
            else:  # no-pad
                seqN_text = encoder_size_text

            encoder_inputs_text = trans[:encoder_size_text]
            encoder_seq_text = seqN_text

            tmp_label = np.zeros(N_CATEGORY, dtype=np.float)
            tmp_label[label] = 1
            labels = tmp_label

            print(encoder_inputs_audio, encoder_seq_audio, encoder_prosody, encoder_inputs_text, encoder_seq_text)

            X = (index, encoder_inputs_text, encoder_inputs_audio)
            Y = labels
            META = (0, 0, 0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])

        else:
            X = (index, self.text[index], self.audio[index])
            Y = self.labels[index]
            META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
            if self.data == 'mosi':
                META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
            if self.data == 'iemocap':
                Y = torch.argmax(Y, dim=-1)
        return X, Y, META

    def load_data(self, split_type):

        if split_type == "valid":
            split_type = "dev"
        audio_mfcc = split_type + "_audio_mfcc.npy"
        mfcc_seqN = split_type + "_audio_seqN.npy"
        audio_prosody = split_type + "_audio_prosody.npy"
        text_trans = split_type + "_nlp_trans.npy"
        label = split_type + "_label.npy"

        print(
            'load data : ' + audio_mfcc + ' ' + mfcc_seqN + ' ' + audio_prosody + text_trans + ' ' + label)
        output_set = []

        for j in range(5):
            print("fold: " + str(j + 1))
            path = os.path.join(self.data_path, "folds", "fold0" + str(j + 1))

            # audio
            tmp_audio_mfcc = np.load(os.path.join(path, audio_mfcc))
            tmp_mfcc_seqN = np.load(os.path.join(path, mfcc_seqN))
            tmp_audio_prosody = np.load(os.path.join(path, audio_prosody))
            tmp_label = np.load(os.path.join(path, label))

            # text
            tmp_text_trans = np.load(os.path.join(path, text_trans))


            for i in range(len(tmp_label)):
                output_set.append(
                    [tmp_audio_mfcc[i], tmp_mfcc_seqN[i], tmp_audio_prosody[i], tmp_text_trans[i], tmp_label[i]])
        print('[completed] load data')

        return output_set
