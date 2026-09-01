from torch.optim import Adam
from dataset import ASVDataset, get_loader, dev_path, dev_protocol, eval_path, eval_protocol
from model import ASVCNN


# Eval loop on dev set
model = ASVCNN()
saved_weights = torch.load(Path("/home/achaammamathayi/Academia/Projects/Attacker fingerprinting/saved_model/trained_model.pth"), weights_only=True)
model.load_state_dict(saved_weights)
model.to(device)
dev_dataloader = get_loader(dev_path, dev_protocol, mfcc_cache, batch_size=128, shuffle=False)
eval_dataloader = get_loader(eval_path, eval_protocol, mfcc_cache, batch_size=128,shuffle=False)
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
print(f'Eval with dev/eval set; Loss: {epoch_loss: .4f}')
print(f'TP: {tp} **** TN: {tn} **** FP: {fp} **** FN: {fn}')
precision = tp/(tp + fp)
print(f'Precision: {precision}')
recall = tp/(tp + fn)
print(f'Recall: {recall}')

