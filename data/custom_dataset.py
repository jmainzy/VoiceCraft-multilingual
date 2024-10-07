import os
from typing import Optional
from datasets.info import DatasetInfo
from datasets.splits import NamedSplit
from datasets.table import Table
import torch
import random
import copy
import logging
import shutil
import glob
from datasets import Audio, Dataset, DatasetDict
import pandas as pd

def load_custom_data(dir):
    '''
    Make an audio dataset dict from a folder
    This returns a Dataset object in HuggingFace Audio format
    '''
    logging.info(f"Reading dataset from {dir}")
    audio_folder = "audio"
    splits = ['train', 'test', 'validation']
    datasets = DatasetDict()
    for split in splits:
        metadata={"audio":[], "duration":[], "segment_id":[], "text":[],"text_latin":[], "file":[]}
        file = os.path.join(dir, split+'.tsv')
        print('reading file '+str(file))
        data = pd.read_csv(file, sep='\t')
        for index, row in data.iterrows():
            metadata["audio"].append(os.path.join(dir, audio_folder, row['filename']))
            metadata["duration"].append(float(row['duration']))
            metadata["segment_id"].append(row['filename'].split('.')[-2])
            metadata["text"].append(row['text'])
            metadata["text_latin"].append(row['latin'])
            metadata["file"].append(file)

        dataset = Dataset.from_dict(metadata).cast_column("audio", Audio(sampling_rate=16_000))
        datasets[split] = dataset
    return datasets


class EncodecDataset(torch.utils.data.Dataset):
    '''
    Dataset in PyTorch format (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
    Simple dataset for generating encoding
    '''
    def __init__(self, split):
        super.__init__()
        self.split = split

    def __len__(self):
        return len(self.data)
    
    @classmethod
    def from_huggingface(cls, split, dataset):
        '''
        Create an instance of the class from a huggingface dataset
        '''
        instance = cls(split=split)
        instance.data = dataset[split]
        return instance

    def __getitem__(self, ind):
        try:
            segment_id, audio, duration, text= \
                    self.data[ind]['segment_id'], \
                    torch.from_numpy(self.data[ind]['audio']['array']).float(), \
                    self.data[ind]['duration'], \
                    self.data[ind]['text']
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None, None
        return segment_id, audio, duration, text
    
    def collate(self, batch):
        res = {'segment_id': [], "audio": [], "duration": [], "text": [], }
        for item in batch:
            if item[0] != None:
                res['segment_id'].append(item[0])
                res['audio'].append(item[1])
                res['duration'].append(item[2])
                res['text'].append(item[3])
        return res

class TTSDataset(torch.utils.data.Dataset):
    '''
    Dataset in PyTorch format (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
    This is for training a VoiceCraft model
    Based off of the Gigaspeech dataset

    args: 
    - dataset_dir: the directory of the dataset
    - exp_dir: the directory of the experiment
    - phn_folder_name: the name of the folder containing the phoneme files
    - encodec_folder_name: the name of the folder containing the encodec files
    - n_codebooks: the number of codebooks
    - n_special: the number of special tokens
    - special_first: whether to add the special tokens first
    - encodec_sr: the sampling rate of the encodec files
    - audio_min_length: the minimum length of the audio
    - audio_max_length: the maximum length of the audio
    - text_max_length: the maximum length of the text
    - pad_x: whether to pad the text
    - sep_special_token: whether to use a special token for padding
    - dynamic_batching: whether to use dynamic batching
    - drop_long: whether to drop long samples

    '''
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        assert self.split in ['train', 'test', 'validation']
        
        self.data = load_custom_data(self.args.dataset_dir)[split]

        print(self.data)
        if len(self.data) == 0:
            raise ValueError("No data found in the dataset in folder "+str(self.args.dataset_dir))
        self.lengths_list = [item['duration'] for item in self.data]

        # phoneme vocabulary
        vocab_fn = os.path.join(self.args.dataset_dir,"vocab.txt")
        shutil.copy(vocab_fn, os.path.join(self.args.exp_dir, "vocab.txt"))
        with open(vocab_fn, "r") as f:
            temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
            self.phn2num = {item[1]:int(item[0]) for item in temp}
        
        self.symbol_set = set(["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"])

        # if args does not have sep_special_token
        if not hasattr(self.args, "sep_special_token"):
            self.args.sep_special_token = False
    
    def __len__(self):
        return len(self.data)
    
    def _load_phn_enc(self, index):
        item = self.data[index]
        pf = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item['segment_id']+".txt")
        ef = os.path.join(self.args.dataset_dir, self.args.encodec_folder_name, item['segment_id']+".txt")
        try:
            with open(pf, "r") as p, open(ef, "r") as e:
                phns = [l.strip() for l in p.readlines()]
                assert len(phns) == 1, phns
                x = [self.phn2num[item] for item in phns[0].split(" ") if item not in self.symbol_set] # drop ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"], as they are not in training set annotation
                encos = [l.strip().split() for k, l in enumerate(e.readlines()) if k < self.args.n_codebooks]
                
                assert len(encos) == self.args.n_codebooks, ef
                if self.args.special_first:
                    y = [[int(n)+self.args.n_special for n in l] for l in encos]
                else:
                    y = [[int(n) for n in l] for l in encos]
        except Exception as e:
            logging.info(f"loading failed for {pf} and {ef}, maybe files don't exist or are corrupted")
            logging.info(f"error message: {e}")
            return [], [[]]

        return x, y

    def __getitem__(self, index):
        x, y = self._load_phn_enc(index)
        x_len, y_len = len(x), len(y[0])

        if x_len == 0 or y_len == 0:
            return {
            "x": None, 
            "x_len": None, 
            "y": None, 
            "y_len": None, 
            "y_mask_interval": None, # index y_mask_interval[1] is the position of start_of_continue token
            "extra_mask_start": None # this is only used in VE1
            }
        while y_len < self.args.encodec_sr*self.args.audio_min_length:
            assert not self.args.dynamic_batching
            index = random.choice(range(len(self))) # regenerate an index
            x, y = self._load_phn_enc(index)
            x_len, y_len = len(x), len(y[0])
        if self.args.drop_long:
            while x_len > self.args.text_max_length or y_len > self.args.encodec_sr*self.args.audio_max_length:
                index = random.choice(range(len(self))) # regenerate an index
                x, y = self._load_phn_enc(index)
                x_len, y_len = len(x), len(y[0])

        ### padding and cropping below ###
        ### padding and cropping below ###
        # adjust the length of encodec codes, pad to max_len or randomly crop
        orig_y_len = copy.copy(y_len)
        max_len = int(self.args.audio_max_length * self.args.encodec_sr)
        if y_len > max_len:
            audio_start = random.choice(range(0, y_len-max_len))
            for i in range(len(y)):
                y[i] = y[i][audio_start:(audio_start+max_len)]
            y_len = max_len
        else:
            audio_start = 0
            if not self.args.dynamic_batching:
                pad = [0] * (max_len - y_len) if self.args.sep_special_token else [self.args.audio_pad_token] * (max_len - y_len)
                for i in range(len(y)):
                    y[i] = y[i] + pad
        
        # adjust text
        # if audio is cropped, and text is longer than max, crop max based on how audio is cropped
        if audio_start > 0 and len(x) > self.args.text_max_length: # if audio is longer than max and text is long than max, start text the way audio started
            x = x[int(len(x)*audio_start/orig_y_len):]
            if len(x) > self.args.text_max_length: # if text is still longer than max, cut the end
                x = x[:self.args.text_max_length]
        
        x_len = len(x)
        if x_len > self.args.text_max_length:
            text_start = random.choice(range(0, x_len - self.args.text_max_length))
            x = x[text_start:text_start+self.args.text_max_length]
            x_len = self.args.text_max_length
        elif self.args.pad_x and x_len <= self.args.text_max_length:
            pad = [0] * (self.args.text_max_length - x_len) if self.args.sep_special_token else [self.args.text_pad_token] * (self.args.text_max_length - x_len)
            x = x + pad
        ### padding and cropping above ###
        ### padding and cropping above ###

        return {
            "x": torch.LongTensor(x), 
            "x_len": x_len, 
            "y": torch.LongTensor(y), 
            "y_len": y_len
            }
            

    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            if item['x'] == None: # deal with load failure
                continue
            for key, val in item.items():
                out[key].append(val)
        res = {}
        if self.args.pad_x:
            res["x"] = torch.stack(out["x"], dim=0)
        else:
            res["x"] = torch.nn.utils.rnn.pad_sequence(out["x"], batch_first=True, padding_value=self.args.text_pad_token)
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.args.dynamic_batching:
            if out['y'][0].ndim==2:
                res['y'] = torch.nn.utils.rnn.pad_sequence([item.transpose(1,0) for item in out['y']],padding_value=self.args.audio_pad_token)
                res['y'] = res['y'].permute(1,2,0) # T B K -> B K T
            else:
                assert out['y'][0].ndim==1, out['y'][0].shape
                res['y'] = torch.nn.utils.rnn.pad_sequence(out['y'], batch_first=True, padding_value=self.args.audio_pad_token)
        else:
            res['y'] = torch.stack(out['y'], dim=0)
        res["y_lens"] = torch.LongTensor(out["y_len"])
        res["text_padding_mask"] = torch.arange(res['x'][0].shape[-1]).unsqueeze(0) >= res['x_lens'].unsqueeze(1)
        res["audio_padding_mask"] = torch.arange(res['y'][0].shape[-1]).unsqueeze(0) >= res['y_lens'].unsqueeze(1)
        return res