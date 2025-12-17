import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.video_cav_mae import VideoCAVMAEFT
import numpy as np
from torch.cuda.amp import autocast
import json
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
import os
import datasets
import torchaudio
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader
from decord import cpu
import torchvision.transforms as T
from src.utilities.stats import calculate_stats
from src.mavosdd_dataset import MavosDD
from src.exddv_dataset import ExDDV
from src.custom_dataset import CustomDDV
import torch
import json
import argparse
import glob
class DeepfakeDetector():
    
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.model = VideoCAVMAEFT()
        self.model = torch.nn.DataParallel(self.model)
        self.model.eval()
        ckpt = torch.load(self.config['checkpoint_path'], map_location='cpu')
        miss, unexp = self.model.load_state_dict(ckpt, strict=False)
        self.skip_norm = self.config.get('skip_norm') if self.config.get('skip_norm') else False
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(size=(self.config['im_res'], self.config['im_res'])),
            T.ToTensor(),   
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )
        ])
    
    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename, backend="ffmpeg")
        waveform = waveform - waveform.mean()

        # try:
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.config['num_mel_bins'], dither=0.0, frame_shift=10)
        # except:
        #     fbank = torch.zeros([512, 128]) + 0.01
        #     print('there is a loading error')

        target_length = self.config['target_length']

        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1,2), size=(target_length, ), mode='linear', align_corners=False).transpose(1,2).squeeze(0)
        return fbank

    def _get_frames(self, video_name):
        vr = VideoReader(video_name)
        total_frames = len(vr) 
        frame_indices = np.linspace(0, total_frames - 1, self.config['num_frames']).astype(int)
        # start_time =time.time()
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [self.preprocess(frame)  for frame in frames]
        
 
            
        return frames


    def analyze_video(self, video_path):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        fbank = self._wav2fbank(video_path)
        frames = self._get_frames(video_path)
        frames = torch.stack(frames)

        freqm = torchaudio.transforms.FrequencyMasking(self.config['freqm'])
        timem = torchaudio.transforms.TimeMasking(self.config['timem'])
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.config['freqm'] != 0:
            fbank = freqm(fbank)
        if self.config['timem'] != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        if self.skip_norm == False:
            fbank = (fbank - self.config['mean']) / (self.config['std'])
        frames = frames.permute(1, 0, 2, 3)
        frames = frames.unsqueeze(0)
        fbank = fbank.unsqueeze(0)

        fbank = fbank.to(device)
        frames = frames.to(device)
        with autocast():
            with torch.no_grad():
                # print(fbank.shape, frames.shape)
                # print(torch.amax(fbank),torch.amin(fbank),torch.amax(frames),torch.amin(frames))
                # exit()
                output = self.model(fbank, frames)
                output = F.softmax(output, dim=-1).cpu().numpy()
        output = {'real_score': output[0][1], 'fake_score': output[0][0]}
        self.model.to("cpu")
        return output
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    args = parser.parse_args()
    df_detector = DeepfakeDetector("configs/config.json")
    if os.path.isdir(args.video_path):
        video_paths = glob.glob(f"{args.video_path}/*.mp4")
        result = []
        for video in video_paths:
            output = df_detector.analyze_video(video)
            output["fake_score"] = str(output["fake_score"])
            output["real_score"] = str(output["real_score"])
            output["video"] = video
            result.append(output)
        with  open('result.json', 'w') as f:
            json.dump(result, f, indent=4)
        print(result)
    else:
        
        output = df_detector.analyze_video(args.video_path)
        output["fake_score"] = str(output["fake_score"])
        output["real_score"] = str(output["real_score"])
        output["video"] = args.video_path
        print(output)
        with  open('result.json', 'w') as f:
            json.dump(output, f, indent=4)
    # df_detector = DeepfakeDetector("configs/config.json")
    # ds_path = "/home/biodeep/alin/datasets/MAVOS-DD"
    # ds = load_from_disk(ds_path).filter(lambda sample: sample['split']=='test' and sample['language']=='romanian' and sample['generative_method']=='inswapper')
    # ds = ds.shuffle(seed=42)[:1000]
    # predictions = []
    # labels = []
    # for i, video in enumerate(tqdm(ds['video_path'])):
    #     video_path = os.path.join(ds_path, video)
    #     labels.append(1 if ds['label'][i] == 'fake' else 0 )
    #     output = df_detector.analyze_video(video_path)
    #     predictions.append(1 if output['real_score']< output['fake_score'] else 0)

    # print("Accuracy: ", np.mean(np.array(predictions)== np.array(labels)))