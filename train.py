from torch.optim import Adam
from dataset import ASVDataset, get_loader, train_path, train_protocol
from model import ASVCNN

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
dataloader = get_loader(train_path,train_protocol, mfcc_cache, batch_size=64, shuffle=True)

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

epochs = 10

for epoch in range(epochs):
    epoch_loss = train_epoch(model, dataloader, criterion, optimizer, device)
    print(f'Epoch: {epoch+1} ==================== Loss: {epoch_loss: .4f}')

# Saving the model, model weights are only saved since  saving the whole model can cause a fail
# in the future if the model code gets changed
torch.save(model.state_dict(), Path("/home/achaammamathayi/Academia/Projects/Attacker fingerprinting/saved_model/trained_model.pth"))
