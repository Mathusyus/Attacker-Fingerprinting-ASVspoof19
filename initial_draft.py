# Continuing ASVspoof anti-spoofing project. Dataset class is complete and working. Moving to CNN model design. Input shape is (64, 1, 20, 125). Using Conv2d. Guide me the same way as before — Socratic method, no code handed directly
# Dayum bro, Claude is ruthless. I learned more speech processing and coding in the last 3 weeks than the whole 2025.
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import torch.nn as nn


train_protocol = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
dev_protocol = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
eval_protocol = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
train_path = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_train/flac")
dev_path = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_dev/flac")
eval_path = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_eval/flac")

# Refined Dataset class
class ASVDataset(Dataset):
    def __init__(self, audio_folder, label_file):
        super().__init__()
        self.audio_folder = audio_folder
        self.label_file = label_file
        self.label_tuple = self.parser()

# Parser function
    def parser(self):
        self.label_tuple = []
        with open(self.label_file, 'r') as f:
            for line in f:
                parts = line.split()
                file_name = parts[1] + '.flac'
                bonorpspoof = 0 if parts[4] == 'bonafide' else 1
                self.label_tuple.append([file_name,bonorpspoof])
        return self.label_tuple
    
    def __len__(self):
        return len(self.label_tuple)
    
    def __getitem__(self, index):
        # Plan: extract audio from the full_path. link it with the label. That should give something like mfcc sample, label.
        audio_path = self.audio_folder.joinpath(self.label_tuple[index][0])
        try:
            m, sr = librosa.load(path=audio_path,sr=16000)
        except FileNotFoundError as e:
            print(f"FILE MISSING: {e}")
            raise
        mfcc = librosa.feature.mfcc(y=m, n_mfcc=20,)
        if mfcc.shape[1] > 125:
            mfcc = mfcc[:, : 125]
        elif mfcc.shape[1] == 125:
            pass
        elif mfcc.shape[1] < 125:
            pad_size = 125 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0), (0, pad_size)), 'constant', constant_values=0)
        x = torch.tensor(mfcc, dtype=torch.float32)
        y= torch.tensor(self.label_tuple[index][1])
        return x,y