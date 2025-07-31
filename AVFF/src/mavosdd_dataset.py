import os
import torch
import torchaudio
import numpy as np
import torchaudio
import datasets
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader
from decord import cpu
import torchvision.transforms as T
import PIL
import csv
import random
import time
from PIL import ImageEnhance
from memory_profiler import profile
import gc
class RandomCropAndResize:
    def __init__(self, im_res):
        self.im_res = im_res

    def __call__(self, x):
        crop = T.RandomCrop(self.im_res)
        resize = T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC)
        return resize(crop(x))

class RandomAdjustContrast:
    def __init__(self, factor: list):
        self.factor = random.uniform(factor[0], factor[1])

    def __call__(self, x):
        return ImageEnhance.Contrast(x).enhance(self.factor)

class RandomColor:
    def __init__(self, factor: list):
        self.factor = random.uniform(factor[0], factor[1])

    def __call__(self, x):
        return ImageEnhance.Color(x).enhance(self.factor)


class MavosDD(Dataset):
    def __init__(self, dataset, input_path, audio_conf, stage, num_frames=16):
        self.num_frames = num_frames
        self.stage = stage

        self.dataset = dataset
        self.input_path = input_path

        print('Dataset has {:d} samples'.format(len(self.dataset)))
        self.num_samples = len(self.dataset)
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(size=(self.im_res, self.im_res)),
            T.ToTensor(),   
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )
        ])

        # self.preprocess_aug = T.Compose([
        #     T.ToPILImage(),
        #     RandomCropAndResize(self.im_res),
        #     RandomAdjustContrast([0.5, 5]),  
        #     RandomColor([0.5, 5]),
        #     T.ToTensor(),   
        #     T.Normalize(
        #         mean=[0.4850, 0.4560, 0.4060],
        #         std=[0.2290, 0.2240, 0.2250]
        #     )
        # ])
        
        # Perform augment
        # For Stage1, we can concat two real videos, clip, flip the video frames
        self.augment_1 = ['None']
        self.augment_1_weight = [5]
        
        # For Stage2, we can concat two real videos, one real video & one fake video, replace with a random audio
        self.augment_2 = ['None', 'concat', 'replace']
        self.augment_2_weight = [5, 1, 1]

    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename, backend="ffmpeg")
        waveform = waveform - waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        target_length = self.target_length
        # n_frames = fbank.shape[0]

        # p = target_length - n_frames

        # # cut and pad
        # if p > 0:
        #     m = torch.nn.ZeroPad2d((0, 0, 0, p))
        #     fbank = m(fbank)
        # elif p < 0:
        #     fbank = fbank[0:target_length, :]

        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1,2), size=(target_length, ), mode='linear', align_corners=False).transpose(1,2).squeeze(0)

        return fbank

    def _concat_wav2fbank(self, filename1, filename2):
        waveform1, sr1 = torchaudio.load(filename1, backend="ffmpeg")
        waveform2, sr2 = torchaudio.load(filename2, backend="ffmpeg")
        waveform1 = waveform1 - waveform1.mean()
        waveform2 = waveform2 - waveform2.mean()

        try:
            fbank1 = torchaudio.compliance.kaldi.fbank(waveform1, htk_compat=True, sample_frequency=sr1, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
            fbank2 = torchaudio.compliance.kaldi.fbank(waveform2, htk_compat=True, sample_frequency=sr2, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank1 = torch.zeros([512, 128]) + 0.01
            fbank2 = torch.zeros([512, 128]) + 0.01
            print("there is a loading error")

        fbank = torch.concat((fbank1, fbank2), dim=0)
        
        target_length = self.target_length

        # Perform Down/Up Sample
        fbank = torch.nn.functional.interpolate(fbank.unsqueeze(0).transpose(1,2), size=(target_length,), mode='linear', align_corners=False).transpose(1,2).squeeze(0)

        return fbank
    # @profile
    def _get_frames(self, video_name):
        try:
            vr = VideoReader(video_name)
            total_frames = len(vr)  # Total number of frames in the video
        
            # Calculate the indices to sample uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            start_time =time.time()
            # Read the frames using the calculated indices
            frames = vr.get_batch(frame_indices).asnumpy()
            frames = [self.preprocess(frame)  for frame in frames]
            print(f"Reading time: {time.time()-start_time}, {len(vr)}")
        except:
            frames = [torch.zeros(3, 224, 224) for i in range(self.num_frames)]
            
        return frames
    # @profile
    def _concat_get_frames(self, video_name1, video_name2):
        try:
            vr1 = VideoReader(video_name1)
            vr2 = VideoReader(video_name2)

            len1 = len(vr1)
            len2 = len(vr2)
            total_frames = len1 + len2

            # Compute indices to sample from the combined video
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

            frames = []

            for idx in frame_indices:
                if idx < len1:
                    frame = vr1[idx].asnumpy()
                else:
                    frame = vr2[idx - len1].asnumpy()
                frames.append(self.preprocess(frame))

            # Optional: explicitly delete and collect memory
            del vr1, vr2
            gc.collect()

        
        except:
            frames = torch.zeros(self.num_frames, 3, 224, 224)
            
        return frames
    
    # @profile
    def _augment_concat(self, index):
        sample = self.dataset[index]
        video_name = os.path.join(self.input_path,sample["video_path"])
        label = 0 if sample["label"] == "real" else 1
        
        index_1 = random.choice([i for i in range(len(self.dataset))])
        sample_1 = self.dataset[index_1]
        video_name_1 = os.path.join(self.input_path,sample_1["video_path"])
        label_1 = 0 if sample_1["label"] == "real" else 1

        fbank = self._concat_wav2fbank(video_name, video_name_1)
        frames = self._concat_get_frames(video_name, video_name_1)

        if self.stage == 1:
            label_ = 0
        else:
            if int(label) == 0 and int(label_1) == 0:
                label_ = 0
            else:
                label_ = 1
        
        return fbank, frames, label_

    def _augment_replace(self, index):
        sample = self.dataset[index]
        video_name = os.path.join(self.input_path,sample["video_path"])
        label = 0 if sample["label"] == "real" else 1
        # if int(label) == 0:
        #     frames = self._get_frames(video_name)
        #     fbank = self._wav2fbank(video_name)
        #     return fbank, frames, label
        # else:
        label = 1
        index_1 = random.choice([i for i in range(len(self.dataset))])
        sample_1 = self.dataset[index_1]
        video_name_1 = os.path.join(self.input_path,sample_1["video_path"])
        label_1 = 0 if sample_1["label"] == "real" else 1

        # Replace audio with other
        frames = self._get_frames(video_name)
        fbank = self._wav2fbank(video_name_1)
        return fbank, frames, label
    
    # @profile
    def __getitem__(self, index):
        show_time = False
        
        if show_time: start_time = time.time()
        sample = self.dataset[index]
        if show_time: print(f"Step 1: ", time.time() - start_time)
        if show_time: start_time = time.time()
        
        video_name = os.path.join(self.input_path,sample["video_path"])
        label = 0 if sample["label"] == "real" else 1

        # Do not perform data augment under eval mode
        if self.mode == 'eval':
            try:
                fbank = self._wav2fbank(video_name)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
            
            frames = self._get_frames(video_name)
            # frames = [self.preprocess(frame) for frame in frames]
            frames = torch.stack(frames)
        
        else:
            # Data Augment
            if self.stage == 1:
                augment = random.choices(self.augment_1, weights=self.augment_1_weight)[0]
            elif self.stage == 2:
                augment = random.choices(self.augment_2, weights=self.augment_2_weight)[0]
            if augment == 'concat':
                fbank, frames, label = self._augment_concat(index)
            elif augment == 'replace':
                fbank, frames, label = self._augment_replace(index)
            else:
                try:
                    fbank = self._wav2fbank(video_name)
                except:
                    fbank = torch.zeros([self.target_length, 128]) + 0.01
                    print('there is an error in loading audio')
                    
                if show_time: print(f"Step 2: ", time.time() - start_time)
                if show_time: start_time = time.time()
                
                frames = self._get_frames(video_name)
                
                if show_time: print(f"Step 3: ", time.time() - start_time)
                if show_time: start_time = time.time()

            # for i, frame in enumerate(frames):
                # if random.uniform(0, 1) < 0.1:
                #     frames[i] = self.preprocess_aug(frame)
                # else:
                #     frames[i] = self.preprocess(frame)
            # frames = [self.preprocess(frame) for frame in frames]
            frames = torch.stack(frames)
            
            if show_time: print(f"Step 4: ", time.time() - start_time)
            if show_time: start_time = time.time()

            # SpecAug, not do for eval set
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)
            
            if show_time: print(f"Step 5: ", time.time() - start_time)
            if show_time: start_time = time.time()

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass
        
        if show_time: print(f"Step 6: ", time.time() - start_time)
        if show_time: start_time = time.time()

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)
            
        if show_time: print(f"Step 7: ", time.time() - start_time)
        if show_time: start_time = time.time()

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # frames: (T, C, H, W) -> (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)
        
        label = torch.tensor([int(label), 1-int(label)]).float()

        return fbank, frames, label, sample["video_path"]

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    audio_conf = {
        'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mode':'train', 
        'mean': -5.081, 'std': 4.4849, 'noise': False, 'label_smooth': 0, 'im_res': 224
    }
    # dataset = MavosDD(input_path="/home/eivor/data/MAVOS-DD", audio_conf=audio_conf, stage=2)
    # dataset[0]

    mavos_dd = datasets.Dataset.load_from_disk("/home/eivor/data/MAVOS-DD")
    
    """
    # Train
    print(f"Train: ", len(mavos_dd.filter(lambda sample: sample['split']=="train")))
    # Val
    print(f"Val: ", len(mavos_dd.filter(lambda sample: sample['split']=="validation")))
    # Test
    print(f"Total test: ", len(mavos_dd.filter(lambda sample: sample['split']=="test")))

    # Test closed_set
    print(f"Test closed-set: ", len(mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==False)))

    # Test open model
    print(f"Test open-model: ", len(mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==True and sample["open_set_language"]==False)))
    # Test open language
    print(f"Test open-language: ", len(mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==False and sample["open_set_language"]==True)))
    # Test open
    print(f"Test open: ", len(mavos_dd.filter(lambda sample: sample['split']=="test" and sample['open_set_model']==True and sample["open_set_language"]==True)))
    """
    
    print(len(mavos_dd))
    
    mavos_dd.filter(lambda sample: sample['split']=="train")
    
    print(len(mavos_dd))
    
    # train_loader = DataLoader(
    #     MavosDD(mavos_dd.filter(lambda sample: sample['split']=="train"), "/home/eivor/data/MAVOS-DD", audio_conf, stage=2),
    #     batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    # )

    # curr_time = time.time()
    # for batch in train_loader:
    #     # print(batch)
    #     print(f"----------------------------Batch time", time.time() - curr_time)
    #     curr_time = time.time()
