import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.mavosdd_dataset import MavosDD
from src.utilities.stats import calculate_stats

import matplotlib
matplotlib.rcParams.update({'font.size': 26})


DATASET_INPUT_PATH = "/home/eivor/data/MAVOS-DD"
CHECKPOINT_PATH = "/home/eivor/biodeep/Detection/OpenAVFF/egs/exp/stage-3/models/best_optim_state.pth"
CHECKPOINT_PATH = "/home/eivor/biodeep/Detection/OpenAVFF/egs/exp/stage-3/models/best_audio_model.pth"


dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}

def plot_confusion_matrix_percent(y_true, y_pred, labels=None, normalize='true', name="name"):
    labels = []
    for y in y_true:
        if y==1:
            labels.append("Real")
        else:
            labels.append("Fake")
    predictions = []
    for y in y_pred:
        if y==1:
            predictions.append("Real")
        else:
            predictions.append("Fake")
    cm = confusion_matrix(labels, predictions, labels=np.unique(labels), normalize=normalize)
    cm_percent = cm * 100 
    print(cm.shape)
    plt.figure(figsize=(5, 5))
    if name !="open-set":
        sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Fake', "Real"], yticklabels=['Fake', "Real"], cbar=False)
    else:
        sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Fake', "Real"], yticklabels=['Fake', "Real"], cbar=False)
    plt.xlabel("Predicted Label", labelpad=20)
    plt.ylabel("True Label")
    # plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"confusion_matrices/{name}-avff.png")
    plt.close()

def evaluate_model(dataset):
    val_loader = torch.utils.data.DataLoader(
        MavosDD(dataset, DATASET_INPUT_PATH, val_audio_conf, stage=2),
        batch_size=32, shuffle=False, num_workers=8, pin_memory=False
    )
    
    A_predictions, A_targets = [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing data"):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                audio_output = audio_model(a_input, v_input)
            # probabilities = torch.sigmoid(audio_output).cpu().numpy()
            
            A_predictions.append(audio_output.to('cpu'))
            A_targets.append(labels.to('cpu'))

        stats = calculate_stats(
            torch.cat(A_predictions).cpu(),
            torch.cat(A_targets).cpu()
        )
        
         
    return stats   
    
if __name__ == "__main__":
    with open("predictions_mavos.json") as input_json_file:
        preds_json = json.load(input_json_file)
        
    mavos_dd = datasets.Dataset.load_from_disk(DATASET_INPUT_PATH)
    
    split_closed_set = mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==False)

    split_to_evaluate = "closed-set"
    split_to_evaluate = "open-model"
    split_to_evaluate = "open-language"
    split_to_evaluate = "open-set"

    if split_to_evaluate == "closed-set":
        # Test closed-set
        curr_split = split_closed_set
    elif split_to_evaluate == "open-model":
        # Open model
        curr_split = datasets.concatenate_datasets([
            split_closed_set,
            mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==True and sample["open_set_language"]==False)
        ])
    elif split_to_evaluate == "open-language":
        # Open language
        curr_split = datasets.concatenate_datasets([
            split_closed_set,
            mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==True)
        ])
    elif split_to_evaluate == "open-set":
        # Open set
        curr_split = mavos_dd.filter(lambda sample: sample['split']=="test")

    y_pred = []
    y_true = []
    for sample in curr_split:
        entry = preds_json[sample["video_path"]]
        y_pred.append(entry["pred"])
        y_true.append(entry["true"])
        
    stats = calculate_stats(torch.tensor(y_pred), torch.tensor(y_true))
    
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy
    
    print(f"{split_to_evaluate}: {mAP=}, {mAUC=}, {acc=}\n")
    
    plot_confusion_matrix_percent(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), name=split_to_evaluate)

"""
Fine-tuned:
Closed-set: mAP=0.9501837501637329, mAUC=0.9481767559252561, acc=0.8693101355452154
Open model: mAP=0.5, mAUC=-1.0, acc=0.6582830058864001
Open language: mAP=0.8560615794926437, mAUC=0.8525392515611991, acc=0.8245014245014245
Open set: mAP=0.5, mAUC=-1.0, acc=0.7689301416707377

Pre-trained:
closed-set: mAP=0.5058405236716984, mAUC=0.5108842637351005, acc=0.5244790612988064
open-model: mAP=0.5000594658391637, mAUC=0.5014786631015057, acc=0.22575869726128794
open-language: mAP=0.5060315050948454, mAUC=0.5112134750722945, acc=0.5946055658292364
open-set: mAP=0.5018749325745602, mAUC=0.5044609013966188, acc=0.3534411110217432
"""
