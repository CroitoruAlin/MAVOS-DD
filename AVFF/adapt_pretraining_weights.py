import torch
from src.models.video_cav_mae import *
from collections import OrderedDict

input_weight_path = "checkpoints/stage-2.pth"
output_weight_path = "checkpoints/stage-3-adapted.pth"

stage1_weight = torch.load(input_weight_path)

cavmae_ft = VideoCAVMAEFT()
cavmae_ft = torch.nn.DataParallel(cavmae_ft)

stage2_weight = OrderedDict()
for k in stage1_weight.keys():
    if ('mlp' in k and ('a2v' in k or 'v2a' in k)) or 'decoder' in k:
        continue
    stage2_weight[k] = stage1_weight[k]

missing, unexpected = cavmae_ft.load_state_dict(stage2_weight, strict=False)
missing, unexpected
torch.save(cavmae_ft.state_dict(), output_weight_path)