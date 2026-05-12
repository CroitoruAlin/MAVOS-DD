import torch
import torch.nn as nn
from src.models.video_cav_mae import VideoCAVMAEFT, VideoCAVMAEAASISTFT
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
import datasets

from src.utilities.stats import calculate_stats
from src.mavosdd_dataset import MavosDD
from src.exddv_dataset import ExDDV
from src.custom_dataset import CustomDDV
from src.fakeavceleb_dataset import FakeAVCeleb
from src.avlips_dataset import AVLips
from src.celeb_df_dataset import CelebDF
from src.vox_celeb_dataset import VoxCeleb
from src.biodeep_dataset import Biodeep
from peft import LoraConfig, get_peft_model, PeftModel
DATASET_INPUT_PATH = "/mnt/data/datasets/MAVOS-DD"
CHECKPOINT_PATH = "checkpoints/stage-3-adapted.pth"
CHECKPOINT_PATH_LORA = "best_audio_model"
mlp_modules = [
        'a2v.mlp.linear',
        'v2a.mlp.linear',
        'mlp_vision',
        'mlp_audio',
        'mlp_head.fc1',
        'mlp_head.fc2',
    ]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
audio_model = VideoCAVMAEFT()#VideoCAVMAEAASISTFT(n_frames=32, audio_length=2048)#VideoCAVMAEAASISTFT(n_frames=16, audio_length=2048)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.eval()
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
miss, unexp = audio_model.load_state_dict(ckpt, strict=False)
assert len(miss) == 0 and len(unexp) == 0
audio_model = PeftModel.from_pretrained(audio_model.module, model_id = CHECKPOINT_PATH_LORA)
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': 224}

    
if __name__ == "__main__":
    audio_model.to(device)
    
    # mavos_dd = datasets.Dataset.load_from_disk(DATASET_INPUT_PATH)
    # old_mavos = datasets.Dataset.load_from_disk("../old_MAVOS")
    # old_videos = set(old_mavos['video_path'])
    fakeavceleb_dataset = FakeAVCeleb("../../datasets/FakeAVCeleb", val_audio_conf, stage=3, num_frames=16)#CelebDF("/mnt/data/datasets/celebdf_v2", val_audio_conf, stage=3)#FakeAVCeleb("/mnt/data/datasets/FakeAVCeleb_preprocessed", val_audio_conf, stage=3, num_frames=16)#AVLips("/mnt/data/datasets/AVLips", val_audio_conf, stage=3, num_frames=16)#VoxCeleb("../../datasets/vox2_mp4_2/dev/mp4", val_audio_conf, stage=3, num_frames=16)#Biodeep("../../datasets/BioDeepAV", val_audio_conf, stage=3, num_frames=16)##CelebDF("/home/galadriel/projects/datasets/celebdf_v2", val_audio_conf, stage=3)#fakeavceleb_dataset = #Biodeep("../../datasets/BioDeepAV", val_audio_conf, stage=3, num_frames=16)#VoxCeleb("../../datasets/vox2_mp4_2/dev/mp4", val_audio_conf, stage=3, num_frames=32)##AVLips("/home/galadriel/projects/datasets/AVLips", val_audio_conf, stage=3)#
    # with open("predictions_old_2.json") as input_json_file:
    #     preds_json = json.load(input_json_file)
    # with open("predictions_delta_eccv.json") as input_json_file:
    #     preds_json_2 = json.load(input_json_file)

    # val_loader = torch.utils.data.DataLoader(
    #         MavosDD(mavos_dd.filter(lambda sample: sample['split']=="test"), DATASET_INPUT_PATH, val_audio_conf, stage=3),# and sample['video_path'] not in preds_json and sample['video_path'] not in preds_json_2), DATASET_INPUT_PATH, val_audio_conf, stage=2),
    #         batch_size=4, shuffle=False, num_workers=12, pin_memory=True
    #     )
    val_loader = torch.utils.data.DataLoader(
            fakeavceleb_dataset,
            batch_size=6, shuffle=False, num_workers=12, pin_memory=False
        )

    # print(fakeavceleb_dataset[0])
    # exit()
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
            # print(data_out)
            # exit()
                
    with  open('predictions_fakeavceleb_pretrained_adapted.json', 'w') as f:
      json.dump(data_out, f, indent=4)
