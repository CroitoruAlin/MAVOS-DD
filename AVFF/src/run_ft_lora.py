import argparse
import os
import torch
import datasets
from torch.utils.data import DataLoader
from dataloader import VideoAudioDataset
from models.video_cav_mae import VideoCAVMAEFT, VideoCAVMAEAASISTFT
import warnings

from mavosdd_dataset import MavosDD
from avlips_dataset import AVLips
from celeb_df_dataset import CelebDF
from exddv_dataset import ExDDV
from torch.utils.data import ConcatDataset, Subset
from faceforensics_dataset import FaceForensics
from fakeavceleb_dataset import FakeAVCeleb
import sys
import os
import datetime
import time
from utilities import *
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train(model, train_loader, test_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(True)
    
    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_meter = AverageMeter()
    
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.save_dir
    
    # if not isinstance(model, torch.nn.DataParallel):
    #     model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    mlp_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(key in name for key in mlp_modules):
                mlp_params.append(param)
            else:
                lora_params.append(param)
    
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam([{'params': lora_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)
    
    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in lora_params) / 1e6))
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    main_metrics = args.metrics
    
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    
    epoch += 1
    scaler = GradScaler()
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])  # for each epoch, 10 metrics to record
    model.train()
    
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        A_predictions = []
        A_targets = []
        start_time = time.time()
        train_loader = DataLoader(
            create_random_balanced_dataset(mavos_dd_train, celebdf, avlips), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
        for i, (a_input, v_input, labels, _) in enumerate(tqdm(train_loader)):
            # print(f"step 1: ", time.time() - start_time)
            # start_time = time.time()
            assert a_input.shape[0] == v_input.shape[0]
            B = a_input.shape[0]
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            labels = labels.to(device)
            
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()
            
            # print(f"step 2: ", time.time() - start_time)
            # start_time = time.time()
            with autocast():
                output = model(a_input, v_input)
                # print(f"step 3: ", time.time() - start_time)
                # start_time = time.time()
                loss = loss_fn(output, labels)
                # print(f"step 4: ", time.time() - start_time)
                # start_time = time.time()
                A_predictions.append(output.to('cpu').detach())
                A_targets.append(labels.to('cpu'))
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # print(f"step 5: ", time.time() - start_time)
            # start_time = time.time()
            
            # loss_av is the main loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                audio_output = torch.cat(A_predictions)
                target = torch.cat(A_targets)
                stats = calculate_stats(audio_output.cpu(), target.cpu())
                mAP = np.mean([stat['AP'] for stat in stats])
                mAUC = np.mean([stat['auc'] for stat in stats])
                acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy

                if main_metrics == 'mAP':
                    print("mAP: {:.6f}".format(mAP))
                else:
                    print("acc: {:.6f}".format(acc))
                print("AUC: {:.6f}".format(mAUC))
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
            
            # print(f"step 6: ", time.time() - start_time)
            # start_time = time.time()
        
        print('start validation')
        stats, valid_loss = validate(model, test_loader, args)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')
        
        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            # torch.save(model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            os.makedirs("%s/models/best_audio_model" % (exp_dir), exist_ok=True)
            model.save_pretrained("%s/models/best_audio_model" % (exp_dir))
            
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            #torch.save(model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
            os.makedirs("%s/models/audio_model_%d" % (exp_dir, epoch), exist_ok=True)
            model.save_pretrained("%s/models/audio_model_%d" % (exp_dir, epoch))
            
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
        else:
            scheduler.step()
            
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_meter.reset()
        
def validate(model, val_loader, args, output_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels, _) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            with autocast():
                audio_output = model(a_input, v_input)

            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            labels = labels.to(device)
            loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)

        stats = calculate_stats(audio_output.cpu(), target.cpu())

    if output_pred == False:
        return stats, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return stats, audio_output, target
def create_random_balanced_dataset(ds1, ds2, ds3):
    # 1. Set seed for reproducibility (optional)
    generator = torch.Generator()
    # 2. Find the minimum length
    min_len = min(len(ds1), len(ds2), len(ds3))
    
    # 3. Create random indices for each dataset
    # randperm(n) returns a random permutation of integers from 0 to n-1
    idx1 = torch.randperm(len(ds1), generator=generator)[:2*min_len]
    idx2 = torch.randperm(len(ds2), generator=generator)[:min_len]
    idx3 = torch.randperm(len(ds3), generator=generator)[:min_len]
    
    # 4. Create Subsets using the random indices
    subset1 = Subset(ds1, idx1.tolist())
    subset2 = Subset(ds2, idx2.tolist())
    subset3 = Subset(ds3, idx3.tolist())
    
    # 5. Combine them
    combined_dataset = ConcatDataset([subset1, subset2, subset3])
    
    print(f"Combined Dataset Size: {len(combined_dataset)} ({min_len} samples from each)")
    return combined_dataset

parser = argparse.ArgumentParser(description='Video CAV-MAE')
parser.add_argument('--input_path', type=str, help='path to data')
parser.add_argument('--target_length', default=1024, type=int, help='audio target length')
parser.add_argument("--dataset_mean", default=-5.081, type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", default=4.4849, type=float, help="the dataset audio spec std, used for input normalization")
parser.add_argument("--noise", default=False, type=bool, help="add noise to the input")

parser.add_argument('--batch-size', default=4, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument('--n-epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--n_classes', default=2, type=int, help='Num of classes to be classified')
parser.add_argument('--save-dir', default='checkpoints', type=str, help='directory to save checkpoints')
parser.add_argument('--pretrain_path', default=None, type=str, help='path to pretrain model')
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
parser.add_argument('--save_model', default=True)
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate')
parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', default=None)
parser.add_argument("--n_print_steps", default=100, type=int)
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument('--warmup',type=bool, default=True)
parser.add_argument('--head_lr', type=int, default=50)

parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")

args = parser.parse_args()

im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mode':'train',
            'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'mode':'eval',
            'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))

# Construct dataloader

input_path = args.input_path
mavos_dd = datasets.Dataset.load_from_disk(input_path)
mavos_dd_train =  MavosDD(mavos_dd.filter(lambda sample: sample['split']=="train"), input_path, audio_conf, stage=3, num_frames=16)
celebdf = CelebDF("/mnt/data/datasets/celebdf_v2_preprocessed", audio_conf, stage=3, num_frames=16)
avlips = AVLips("/mnt/data/datasets/AVLips_preprocessed", audio_conf, stage=3, num_frames=16)
fakeavceleb_dataset = FakeAVCeleb("../../datasets/FakeAVCeleb", audio_conf, stage=3, num_frames=16)
faceforensics = FaceForensics("/mnt/data/datasets/faceforensics", audio_conf, stage=3, num_frames=16)
final_ds = create_random_balanced_dataset(mavos_dd_train, celebdf, avlips)
train_loader = DataLoader(
   final_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
)
val_ds = MavosDD(mavos_dd.filter(lambda sample: sample['split']=="validation"), input_path, audio_conf, stage=3, num_frames=16)
val_loader = DataLoader(
    fakeavceleb_dataset,
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False
)

print(f"Using Train: {len(train_loader)}, Eval: {len(val_loader)}")

# Construct model
cavmae_ft = VideoCAVMAEFT(n_frames=16, audio_length=1024)#VideoCAVMAEFT(n_frames=16, audio_length=1024)#VideoCAVMAEAASISTFT(n_frames=16, audio_length=2048)#VideoCAVMAEFT()

# init model
if args.pretrain_path is not None:
    mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
    # if not isinstance(cavmae_ft, torch.nn.DataParallel):
        # cavmae_ft = torch.nn.DataParallel(cavmae_ft)
    new_dict = {}
    for key in mdl_weight:
        new_dict[key[7:]] = mdl_weight[key]
    miss, unexpected = cavmae_ft.load_state_dict(new_dict, strict=False)
    print("Missing: ", miss)
    print("Unexpected: ", unexpected)
    print('now load pretrain model from {:s}, missing keys: {:d}, unexpected keys: {:d}'.format(args.pretrain_path, len(miss), len(unexpected)))
    # for param in cavmae_ft.parameters():
    #     param.requires_grad=False
    # for param in cavmae_ft.aasist.parameters():
    #     param.requires_grad=True
    
else:
    warnings.warn("Note you are finetuning a model without any finetuning.")
from peft import LoraConfig, get_peft_model
mlp_modules = [
        'a2v.mlp.linear',
        'v2a.mlp.linear',
        'mlp_vision',
        'mlp_audio',
        'mlp_head.fc1',
        'mlp_head.fc2',
    ]
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules = ['qkv', 'proj'], lora_dropout=0.05, bias='none', modules_to_save=mlp_modules)
cavmae_ft = get_peft_model(cavmae_ft, lora_config)
print("\n Creating experiment directory: %s"%args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Train model
print("Now start training for %d epochs"%args.n_epochs)
train(cavmae_ft, train_loader, val_loader, args)
