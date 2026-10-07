import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path
import torch.nn as nn
from torch.optim import Adam
import scipy.fftpack as fftpack
from scipy.interpolate import interp1d

source_path = Path("/resources/speech/corpora/ASVSpoof19/LA")
train_protocol = source_path/"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
dev_protocol = source_path/"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
eval_protocol = source_path/"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
train_path = source_path/"ASVspoof2019_LA_train/flac"
dev_path = source_path/"ASVspoof2019_LA_dev/flac"
eval_path = source_path/"ASVspoof2019_LA_eval/flac"

# Setting the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Draft3 Dataset class
class ASVDataset(Dataset):
    def __init__(self, audio_folder, label_file, feature_type, mfcc_cache= Path("mfcc_cache"), cqcc_cache= Path("cqcc_cache")):
        super().__init__()
        self.audio_folder = audio_folder
        self.label_file = label_file
        self.label_tuple = self.parser()
        self.mfcc_cache = mfcc_cache
        self.mfcc_cache.mkdir(parents=True,exist_ok=True)
        self.cqcc_cache = cqcc_cache
        self.cqcc_cache.mkdir(parents=True,exist_ok=True)
        self.feature_type = feature_type

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

# Creating a cache folder for cqcc
    def cqcc_cache_gen(self,idx):
        audio_path = self.audio_folder.joinpath(self.label_tuple[idx][0])
        filename = f"cqcc_{Path(self.label_tuple[idx][0]).stem}.npy"
        cache_path = self.cqcc_cache.joinpath(filename)
        if not cache_path.exists():

            # to extract cqcc from cqt
            c, sr = librosa.load(path=audio_path, sr=16000)
            if len(c) < 16000:
                c = librosa.util.fix_length(c, size=16000)

            # loadin cqt
            cqt = librosa.cqt(c,sr=16000, n_bins=96, bins_per_octave=12, fmin=30)

            # absolute magnutde of cqt
            cqt_abs = np.abs(cqt)

            # log power of cqt_abs
            cqt_log_power = 10 * np.log10(cqt_abs**2 + 1e-10) #episolon to preven log(0)

            # uniform resampling to reform the geomatrically spaced frequency rows into a linear grid
            # this is the messy part
            # first we assign the number of bins and number of target time frames to respective variables
            num_cqt_bins, num_time_frames = cqt_log_power.shape

            # then we define the logarithmic frequency cordinates
            og_freq_axis = np.logspace(0, 1, num=num_cqt_bins) # logarthmic band from 0 to 1 with 'num' parts

            # we assign the target bins pre-hand
            target_bins = 512
            # we then make it into linearly spaced frequeny cordinates
            linear_freq_axis = np.linspace(og_freq_axis[0], og_freq_axis[-1],num=target_bins)

            #creating a oneline 1d interpolation function along axis 0 aka freq axis(y axis has the log power magnitude)
            interpolator = interp1d(og_freq_axis, cqt_log_power, axis=0, kind='linear') # args: original uneven x values, y values and axis=0 tells model to calculate along the frequency axis

            #calling the one line function on the linear freq axis
            cqt_resampled = interpolator(linear_freq_axis)

            cqcc = fftpack.dct(cqt_resampled, type=2, axis=0, norm='ortho')[:96, :]
            if cqcc.shape[1] > 125:
                cqcc = cqcc[:, : 125]
            elif cqcc.shape[1] == 125:
                pass
            elif cqcc.shape[1] < 125:
                pad_size = 125 - cqcc.shape[1]
                cqcc = np.pad(cqcc, ((0,0), (0, pad_size)), 'constant', constant_values=0)
            np.save(cache_path, cqcc)
        else:
            cqcc = np.load(cache_path)

        return cqcc

    def __len__(self):
        return len(self.label_tuple)
    
    def __getitem__(self, index):
        if self.feature_type == 'mfcc':
            x = torch.tensor(self.mfcc_cache_gen(index), dtype=torch.float32)
        elif self.feature_type == 'cqcc':
            x = torch.tensor(self.cqcc_cache_gen(index), dtype=torch.float32)
        y= torch.tensor(self.label_tuple[index][1])
        return x,y


# DataLoader class
temp_folder = Path("/mount/studenten-temp1/users/georgems")
train_dataset = ASVDataset(train_path,train_protocol, feature_type='cqcc', cqcc_cache=temp_folder/"train_cqcc")
dev_dataset = ASVDataset(dev_path, dev_protocol, feature_type='cqcc', cqcc_cache=temp_folder/"dev_cqcc")
eval_dataset = ASVDataset(eval_path, eval_protocol, feature_type='cqcc', cqcc_cache=temp_folder/"test_cqcc")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False, drop_last=False)
eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, drop_last=False)



# Model design
class ASVCNN(nn.Module):
    def __init__(self, input_size=1, num_classes = 1):
        super(ASVCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # might have to change the pooling kernel stride to an overlapping one after the training
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layer2 = nn.Sequential(
            #kernel size remains the same, hence the padding and striding as well
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #changing the pooling kernel size so that the layer 1 feautremap are made in a higher resolution feature map.
            #no pooling around mfcc axis, only pooling along time frame axis
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        #classification layers
        self.linear1 = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        # print('layer1: ',x.shape)
        x = self.layer2(x)
        # print('layer 2',x.shape)
        x = self.layer3(x)
        # print('layer 3',x.shape)

        #flattening x
        x = torch.flatten(x, start_dim=1)
        # print('after flattening',x.shape)

        x = self.linear1(x)
        # print('linear 1:', x.shape)
        x = self.linear2(x)
        # print('linear 2:',x.shape)
        return x


# the training loop
def train_epoch(model, dataloader, criterion, optimizer, device):

    running_loss = 0
    model.train()

    for sample, label in dataloader:
        sample = sample.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(sample)
        label = (label.view(-1,1)).float()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss/len(dataloader)

model = ASVCNN().to(device)
a,b = next(iter(train_loader))
c = torch.randn(1,a.shape[1],125).to(device)
model(c)
dataloader = train_loader

# Calculating the total negatives/positives for pos_weight argument in BCEWithLogitsLoss()
bonfi = 0
spoof = 0
for i in train_dataset.label_tuple:
    if i[1] == 1:
        bonfi += 1
    else:
        spoof += 1
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(spoof/bonfi).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)

epochs = 30

for epoch in range(epochs):
    epoch_loss = train_epoch(model, dataloader, criterion, optimizer, device)
    print(f'Epoch: {epoch+1} ==================== Loss: {epoch_loss: .4f}')

# Saving the model, model weights are only saved since  saving the whole model can cause a fail
# in the future if the model code gets changed
model_folder = Path("/mount/studenten-temp1/users/georgems/model_folder")
model_folder.mkdir(parents=True,exist_ok=True)
torch.save(model.state_dict(), model_folder/"trained_model_cqcc")


# Eval loop on dev set
model = ASVCNN()
saved_weights = torch.load(model_folder/"trained_model_cqcc",weights_only=True)
model.load_state_dict(saved_weights)
model.to(device)
dataloader = dev_loader
criterion = nn.BCEWithLogitsLoss()

def eval_model(model, dataloader, criterion,device):

    running_loss = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    model.eval()

    with torch.no_grad():
        for sample, label in dataloader:
            sample = sample.to(device)
            label = label.to(device)

            output = model(sample)
            label = (label.view(-1,1)).float()
            loss = criterion(output, label)
            predictions = output.sigmoid()
            threshold = 0.7
            predictions_boolean = predictions > threshold

            # True positives = is a bonafide, detected as a bonafide.
            tp_mask = (predictions_boolean.int() == 1) & (label == 1)
            tp += tp_mask.sum().item()

            # True negatives = is a spoof, detected as a spoof
            tn_mask = (predictions_boolean.int() == 0) & (label == 0)
            tn += tn_mask.sum().item()

            # False positives = is a spoof, detected as a bonafide.
            fp_mask = (predictions_boolean.int() == 1) & (label == 0)
            fp += fp_mask.sum().item()

            # False negatives = is a bonafide, detected as a spoof
            fn_mask = (predictions_boolean.int()== 0) & (label == 1)
            fn += fn_mask.sum().item()

            running_loss += loss.item()

        return running_loss/len(dataloader), tp, tn, fp, fn


epoch_loss, tp, tn, fp, fn = eval_model(model, dataloader, criterion, device)
print(f'Eval on dev set; Loss: {epoch_loss: .4f}')
print(f'TP: {tp} **** TN: {tn} **** FP: {fp} **** FN: {fn}')
precision = tp/(tp + fp)
print(f'Precision: {precision}')
recall = tp/(tp + fn)
print(f'Recall: {recall}')



# Eval loop on eval set
model = ASVCNN()
saved_weights = torch.load(model_folder/"trained_model_cqcc",weights_only=True)
model.load_state_dict(saved_weights)
model.to(device)
dataloader = eval_loader
criterion = nn.BCEWithLogitsLoss()

def eval_model(model, dataloader, criterion,device):

    running_loss = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    model.eval()

    with torch.no_grad():
        for sample, label in dataloader:
            sample = sample.to(device)
            label = label.to(device)

            output = model(sample)
            label = (label.view(-1,1)).float()
            loss = criterion(output, label)
            predictions = output.sigmoid()
            threshold = 0.7
            predictions_boolean = predictions > threshold

            # True positives = is a bonafide, detected as a bonafide.
            tp_mask = (predictions_boolean.int() == 1) & (label == 1)
            tp += tp_mask.sum().item()

            # True negatives = is a spoof, detected as a spoof
            tn_mask = (predictions_boolean.int() == 0) & (label == 0)
            tn += tn_mask.sum().item()

            # False positives = is a spoof, detected as a bonafide.
            fp_mask = (predictions_boolean.int() == 1) & (label == 0)
            fp += fp_mask.sum().item()

            # False negatives = is a bonafide, detected as a spoof
            fn_mask = (predictions_boolean.int()== 0) & (label == 1)
            fn += fn_mask.sum().item()

            running_loss += loss.item()

        return running_loss/len(dataloader), tp, tn, fp, fn


epoch_loss, tp, tn, fp, fn = eval_model(model, dataloader, criterion, device)
print(f'Eval on eval set; Loss: {epoch_loss: .4f}')
print(f'TP: {tp} **** TN: {tn} **** FP: {fp} **** FN: {fn}')
precision = tp/(tp + fp)
print(f'Precision: {precision}')
recall = tp/(tp + fn)
print(f'Recall: {recall}')
    