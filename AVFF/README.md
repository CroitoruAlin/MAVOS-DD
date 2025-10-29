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

Use `inference.py` and set the global variables to their correct values (i.e. actual path to the dataset and actual path to the fine-tuned model).

## Performance metrics

To compute the performance metrics use the `eval.py` script.