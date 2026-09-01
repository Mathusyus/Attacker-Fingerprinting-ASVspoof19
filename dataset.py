import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path

train_protocol = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
dev_protocol = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
eval_protocol = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
train_path = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_train/flac")
dev_path = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_dev/flac")
eval_path = Path(r"/home/achaammamathayi/Academia/Projects/Datasets/ASVSpoof19/LA/ASVspoof2019_LA_eval/flac")

# Setting the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Draft3 Dataset class
class ASVDataset(Dataset):
    def __init__(self, audio_folder, label_file, mfcc_cache = Path("mfcc_cache")):
        super().__init__()
        self.audio_folder = audio_folder
        self.label_file = label_file
        self.label_tuple = self.parser()
        self.mfcc_cache = mfcc_cache
        self.mfcc_cache.mkdir(parents=True,exist_ok=True)

# Parser function
    def parser(self):
        self.label_tuple = []
        with open(self.label_file, 'r') as f:
            for line in f:
                parts = line.split()
                file_name = parts[1] + '.flac'
                bonorpspoof = 1 if parts[4] == 'bonafide' else 0
                if self.audio_folder.joinpath(file_name).exists():
                    self.label_tuple.append([file_name,bonorpspoof])
                else:
                     print(f'Missing file: {self.audio_folder.joinpath(file_name)}')
                     continue
        return self.label_tuple

# Creating a cache folder to be in limits of running the code in Uni HPC cluster
    def mfcc_cache_gen(self,idx):
        # Plan: extract audio from the full_path. link it with the label. That should give something like mfcc sample, label.
        audio_path = self.audio_folder.joinpath(self.label_tuple[idx][0])
        filename = f"mfcc_{Path(self.label_tuple[idx][0]).stem}.npy"
        cache_path = self.mfcc_cache.joinpath(filename)
        if not cache_path.exists():
            m, sr = librosa.load(path=audio_path,sr=16000)
            mfcc = librosa.feature.mfcc(y=m, n_mfcc=20,)
            if mfcc.shape[1] > 125:
                mfcc = mfcc[:, : 125]
            elif mfcc.shape[1] == 125:
                pass
            elif mfcc.shape[1] < 125:
                pad_size = 125 - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0,0), (0, pad_size)), 'constant', constant_values=0)
            np.save(cache_path, mfcc)
        else:
            mfcc = np.load(cache_path)

        return mfcc
  
    
    def __len__(self):
        return len(self.label_tuple)
    
    def __getitem__(self, index):
        x = torch.tensor(self.mfcc_cache_gen(index), dtype=torch.float32)
        y= torch.tensor(self.label_tuple[index][1])
        return x,y


# DataLoader class
# train_dataset = ASVDataset(train_path,train_protocol)
# dev_dataset = ASVDataset(dev_path, dev_protocol)
# eval_dataset = ASVDataset(eval_path, eval_protocol)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
# dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False, drop_last=False)
# eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, drop_last=False)

# writing fetcher/helper function to allow varying use across executable modules.

def get_loader(datalocation, labelfile, mfcc_cache, batch_size, shuffle):
    dataset = ASVDataset(datalocation, labelfile, mfcc_cache)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

