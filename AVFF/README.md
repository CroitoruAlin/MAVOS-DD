# Open-AVFF

## Install
First create a conda virtual environment:
```
conda create -n AVFF python=3.9 -y
conda activate AVFF
```
then run the `pip install -r requirements.txt` script, which will install the following modules for you:
```
decord
einops
jupyter
matplotlib
opencv-python
pillow
timm==0.4.5
tqdm
scipy
scikit-learn
```
then install torch: 
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
## Open-AVFF Weights
Download the weights from: https://huggingface.co/acroitoru/avff_mavos
## Inference

Use `inference_single_video.py` and set the checkpoint path in `configs/config.json`.

Usage examples:
```
python ./inference_single_video.py --video_path assets/real
python ./inference_single_video.py --video_path assets/real/9rjQ5sfeUTg_out_151_2.mp4
```
Each comand will generate a `result.json` file.
