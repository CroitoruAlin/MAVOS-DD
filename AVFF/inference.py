import os

import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
import datasets

from src.utilities.stats import calculate_stats
from src.mavosdd_dataset import MavosDD, BioDeepAV
from src.exddv_dataset import ExDDV
from src.custom_dataset import CustomDDV
from src.fakeavceleb_dataset import FakeAVCeleb
from src.models.effort_detector import apply_svd_residual_to_self_attn


DATASET_INPUT_PATH = "/home/eivor/data/PolyGlotFake/BioDeepAV"
# CHECKPOINT_PATH = "/home/eivor/biodeep/Detection/MAVOS-DD/checkpoints/finetuned/avff_mavos.pth"
CHECKPOINT_PATH = "/home/eivor/biodeep/Detection/MAVOS-DD/checkpoints/pretrained/stage-3.pth"

languages_used = "no_encoders"
CHECKPOINT_PATH = f"/home/eivor/biodeep/Detection/MAVOS-DD/AVFF/checkpoints_trained/{languages_used}/models/best_audio_model.pth"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
audio_model = VideoCAVMAEFT()
# apply_svd_residual_to_self_attn(audio_model, r=523)

# audio_model = torch.nn.DataParallel(audio_model)
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
new_ckpt = {}
for k, v in ckpt.items():
    new_key = k.replace("module.", "")
    new_ckpt[new_key] = v
miss, unexp = audio_model.load_state_dict(new_ckpt, strict=False)
audio_model.eval()

print("Missing keys: ", miss)
print("Unexpected keys: ", unexp)
assert len(miss) == 0 and len(unexp) == 0 

dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}

    
if __name__ == "__main__":
    audio_model.to(device)
    
    # MAVOS dataset
    # metadata_indomain = metadata.filter(lambda sample: sample['split']=='test' and not sample['open_set_model'] and not sample['open_set_language'])
    # metadata_open_model = concatenate_datasets([metadata_indomain, metadata_open_model])
    # metadata_open_language = metadata.filter(lambda sample: sample['split']=='test' and not sample['open_set_model'] and sample['open_set_language'])
    # metadata_open_model = concatenate_datasets([metadata_indomain, metadata_open_language])
    # metadata_all = metadata.filter(lambda sample: sample['split']=='test')

    # mavos_dd = datasets.Dataset.load_from_disk("/home/eivor/data/MAVOS-DD")
    # val_loader = torch.utils.data.DataLoader(
    #     MavosDD(
    #         # datasets.concatenate_datasets([
    #         #     mavos_dd.filter(lambda sample: sample['split']=='test' and not sample['open_set_model'] and not sample['open_set_language']),
    #         #     mavos_dd.filter(lambda sample: sample['split']=='test' and not sample['open_set_model'] and sample['open_set_language'])
    #         # ]),
    #         mavos_dd.filter(lambda sample: sample['split']=='test'),
    #         "/home/eivor/data/MAVOS-DD", val_audio_conf, stage=2
    #     ),
    #     batch_size=4, shuffle=False, num_workers=24, pin_memory=False
    # )
    
    # FakeAVCEleb
    fakeavceleb_ds = FakeAVCeleb("/home/eivor/data/FakeAVCeleb_v1.2", val_audio_conf, stage=2)
    val_loader = torch.utils.data.DataLoader(fakeavceleb_ds, batch_size=4, shuffle=False, num_workers=12, pin_memory=False)
    
    # biodeepav_ds = []
    # for label in ["real", "fake"]:
    #     for filename in os.listdir(os.path.join(DATASET_INPUT_PATH, label, "videos")):
    #         if filename.endswith(".mp4"):
    #             biodeepav_ds.append({
    #                 "video_path": os.path.join(label, "videos", filename),
    #                 "label": label
    #             })

    # val_loader = torch.utils.data.DataLoader(
    #         BioDeepAV(biodeepav_ds, DATASET_INPUT_PATH, val_audio_conf, stage=2),
    #         batch_size=4, shuffle=False, num_workers=12, pin_memory=False
    #     )

    
    A_predictions, A_targets = [], []
    data_out = {}
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_paths) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing data"):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                audio_output = audio_model(a_input, v_input).cpu().numpy()
            # probabilities = torch.sigmoid(audio_output).cpu().numpy()
            
            for y_pred,y_true,video_path in zip(audio_output,labels.numpy(),video_paths):
                data_out[video_path] = {
                    "pred": y_pred.tolist(),
                    "true": y_true.tolist(),
                }
                
    with  open(f'predictions_no_encoders_fakeavceleb.json', 'w') as f:
      json.dump(data_out, f, indent=4)
