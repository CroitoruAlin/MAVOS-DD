import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import src.dataloader as dataloader
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
import csv
import datasets

from src.utilities.stats import calculate_stats
from src.mavosdd_dataset import MavosDD


DATASET_INPUT_PATH = "/home/eivor/data/MAVOS-DD"
CHECKPOINT_PATH = "/home/eivor/biodeep/Detection/OpenAVFF/egs/exp/stage-3/models/best_audio_model.pth"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
audio_model = VideoCAVMAEFT()
audio_model = torch.nn.DataParallel(audio_model)
audio_model.eval()
# ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
ckpt = torch.load("/home/eivor/biodeep/Detection/OpenAVFF/checkpoints/stage-3.pth", map_location='cpu')
miss, unexp = audio_model.load_state_dict(ckpt, strict=False)

dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}

def evaluate_model(dataset):
    val_loader = torch.utils.data.DataLoader(
        MavosDD(dataset, DATASET_INPUT_PATH, val_audio_conf, stage=2),
        batch_size=8, shuffle=False, num_workers=8, pin_memory=False
    )
    
    A_predictions, A_targets = [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels, _) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing data"):
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
    audio_model.to(device)
    
    mavos_dd = datasets.Dataset.load_from_disk(DATASET_INPUT_PATH)

    split_to_evaluate = "closed-set"
    # split_to_evaluate = "open-model"
    # split_to_evaluate = "open-language"
    # split_to_evaluate = "open-set"

    if split_to_evaluate == "closed-set":
        # Test closed-set
        curr_split = mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==False)
        stats = evaluate_model(curr_split)
        
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy
        
        print(f"Closed-set: {mAP=}, {mAUC=}, {acc=}\n")

    elif split_to_evaluate == "open-model":
        # Open model
        curr_split = mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==True and sample["open_set_language"]==False)
        stats = evaluate_model(curr_split)
        
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy
        
        print(f"Open model: {mAP=}, {mAUC=}, {acc=}\n")

    elif split_to_evaluate == "open-language":
        # Open language
        curr_split = mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==True)
        stats = evaluate_model(curr_split)
        
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy
        
        print(f"Open language: {mAP=}, {mAUC=}, {acc=}\n")
    
    elif split_to_evaluate == "open-set":
        # Open set
        curr_split = mavos_dd.filter(lambda sample: sample['split']=="test")
        stats = evaluate_model(curr_split)
        
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy
        
        print(f"Open set: {mAP=}, {mAUC=}, {acc=}\n")

"""
Closed-set: mAP=0.9501837501637329, mAUC=0.9481767559252561, acc=0.8693101355452154
Open model: mAP=0.5, mAUC=-1.0, acc=0.6582830058864001
Open language: mAP=0.8560615794926437, mAUC=0.8525392515611991, acc=0.8245014245014245
Open set: mAP=0.5, mAUC=-1.0, acc=0.7689301416707377


"""
