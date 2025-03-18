import os
from argparse import ArgumentParser
import copy

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm

from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler

import wandb
from internvl.internvl import load_internvl, get_ret, get_data_item, get_model_chat, crop_tensor_image
from PIL import Image, ImageDraw

from PIL import Image, ImageDraw
from torchvision import transforms

from torchvision.utils import save_image 
import cv2 
import numpy as np 
import string
from torchvision.transforms.functional import crop
import torch.nn.functional as F 

voc = list(string.printable[:-6])
CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',
                '0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@',
                'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
                'R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b',
                'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
                't','u','v','w','x','y','z','{','|','}','~']

def _decode_recognition(rec):
    s = ''
    for c in rec:
        if c<95:
            s += CTLABELS[c]
        else:
            return s
    return s

# JLP
def decode(rec):
    s = ''
    for c in rec:
        if c<94:
            s += voc[c]
        else:
            return s
    return s

def encode(rec):
    rec = rec.replace(' ','')
    s = []
    for i in range(25):
        if i < len(rec):
            try:
                idx = voc.index(rec[i])
            except:
                if rec[i] == '-':
                    idx = 74 
            s.append(idx)
        elif i == len(rec):
            s.append(94)
        else:
            s.append(95)
    return s


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(42, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    
    # torch.cuda.set_device(1)
    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    # breakpoint()
    # Start From Here for finetuning with OCR
    if cfg.train.resume: 
        # load only controlnet weights 
        # actually diffbir only tuned controlnet weights 
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
            )
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )

    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        print(f"load SwinIR from {cfg.train.swinir_path}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # set trainable parameters
    model_names=[]
    for name, param in cldm.named_parameters():
        model_names.append(name)

        if cfg.pho_args.fine_tuning_method == 'full_fine_tuning':
            param.requires_grad = True 


        elif cfg.pho_args.fine_tuning_method == 'only_sd_fine_tuning':
            if 'unet' in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False


        elif cfg.pho_args.fine_tuning_method == 'only_ctrlnet_fine_tuning':
            if 'controlnet' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


        elif cfg.pho_args.fine_tuning_method == 'etc':
            pass 

    # JLP 
    from adet.config import get_cfg
    def setup(args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        cfg.merge_from_file(args.bridge_config)
        cfg.freeze()
        return cfg
    bridge_config = setup(args)

    # JLP - get spotter model
    from adet.modeling.TESTR import TransformerDetector 
    bridge_model = TransformerDetector(bridge_config)  

    # JLP - get only recognizer model from bridge 
    from DiG.models.model_builder import RecModel
    recognizer = RecModel(bridge_config)
    checkpoint = torch.load("./pretrained_weights/checkpoint-9.pth", map_location='cpu')
    msg=recognizer.load_state_dict(checkpoint["model"], False)
    recognizer.to('cuda')
    # recognizer.eval()

    for param in recognizer.parameters():
        param.requires_grad=False

    print(msg)
    
    # JLP - criterion for recognizer
    from DiG.loss import SeqCrossEntropyLoss
    rec_criterion = SeqCrossEntropyLoss()

    # Setup optimizer:
    # params_to_optimize = [
    #     {"params": cldm.controlnet.parameters(), "lr": cfg.train_settings.learning_rate},
    #     {"params": unet_params, "lr": cfg.train_settings.learning_rate},
    # ]
    # opt = torch.optim.AdamW(params_to_optimize, lr=cfg.train_settings.learning_rate)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, cldm.parameters()), lr=cfg.train_settings.learning_rate)

    # Setup data:
    train_set = instantiate_from_config(cfg.dataset.train)
    val_set = instantiate_from_config(cfg.dataset.val)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.train_settings.train_batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.val.val_batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(train_set):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # internvl, tokenizer = load_internvl()
    
    # Prepare models for training:
    cldm.train().to(device)
    swinir.eval().to(device)
    diffusion.to(device)
    # cldm, opt, train_loader, val_loader, internvl = accelerator.prepare(cldm, opt, train_loader, val_loader, internvl)
    cldm, opt, train_loader, val_loader = accelerator.prepare(cldm, opt, train_loader, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train_settings.train_steps
    step_loss = []
    step_ocr_rec_loss = []  # JLP - additional loss 
    step_total_loss = []
    epoch = 0
    epoch_total_loss = []
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )

    if accelerator.is_main_process:
        wandb.login(key=cfg.wandb_args.wandb_key)
        wandb.init(project="DiffBIR_OCR_finetuning", name=f"{cfg.wandb_args.wandb_exp_name}", config=dict(cfg))
        print(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(train_loader),
        )
        for train_iter, batch in enumerate(train_loader):

            opt.zero_grad()
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt, text, bbox, img_name = batch

            bs = gt.shape[0]

            # JLP - get text encodings and text_lens on the fly (using char level)
            txt_encs=[]
            txt_lens=[]
            for i in range(bs):
                txt = text[i]
                txt_lens.append(len(txt))
                # print(f'text: ', txt)
                txt_enc = encode(txt)
                # print(f'encoded text: ', txt_enc)
                # print('decoded txt: ', decode(txt_enc))
                txt_encs.append(torch.tensor(txt_enc))
            txt_encs = torch.stack(txt_encs)
            txt_lens = torch.tensor(txt_lens)

            # JLP - re organize bbox format 
            xs, ys, ws, hs = bbox[0], bbox[1], bbox[2], bbox[3] 
            boxes = []
            for i in range(bs):
                # scale bbox coordinates 512 -> 448
                x_scale_ratio = 448/512
                y_scale_ratio = 448/512
                x, y, w, h = int(xs[i].item() * x_scale_ratio), int(ys[i].item() * y_scale_ratio), int(ws[i].item() * x_scale_ratio), int(hs[i].item() * y_scale_ratio)
                box = torch.tensor([x, y, x+w, y+h])
                boxes.append(box)
            boxes = torch.stack(boxes)  # x1,y1,x2,y2 format

            # # JLP - vis batch data: img, bbox, text
            # for i in range(bs):
            #     hr = gt[i]      # 448, 448, 3
            #     lr = lq[i]
            #     hr = (hr-hr.min())/(hr.max()-hr.min())*255.0
            #     lr = (lr-lr.min())/(lr.max()-lr.min())*255.0
            #     hr = hr.detach().cpu().numpy().astype(np.uint8)
            #     lr = lr.detach().cpu().numpy().astype(np.uint8)
            #     hr = hr.copy()
            #     lr = lr.copy()
            #     # cv2.imwrite(f'img{i}_hr.jpg', hr[...,::-1])
            #     # cv2.imwrite(f'img{i}_lr.jpg', lr[...,::-1])]
            #     box = boxes[i]
            #     x1, y1, x2, y2 = boxes[i]
            #     x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())

            #     cv2.rectangle(hr, (x1,y1), (x2,y2), color=(0,255,0), thickness=4)
            #     cv2.rectangle(lr, (x1,y1), (x2,y2), color=(0,255,0), thickness=4)
            #     cv2.imwrite(f'img{i}_hr_{text[i]}.jpg', hr[...,::-1])
            #     cv2.imwrite(f'img{i}_lr_{text[i]}.jpg', lr[...,::-1])
            
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()   # b 3 448 448
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()   # b 3 448 448

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)                      # b 4 56 56 
                clean = swinir(lq)                                  # b 3 448 448
                cond = pure_cldm.prepare_condition(clean, prompt)   # cond['c_img']: b 4 56 56
                # noise augmentation
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:  # f
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )
            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )
            # breakpoint()
            loss, pred_z_0, z_t = diffusion.p_losses(cldm, z_0, t, cond_aug)

            '''
                z_0: encoded latent from clean x_0
                z_t: noise added encoded latent
                pred_z_0: using model predicted noise, we can obtain pred_z_0
            '''
            # # JLP - vis importance of amount of noise added, as it can affect pred_x_0 img quality (especially text region)
            # # which is directly connected to OCR quality
            # for i in range(0,1000,200):

            #     t = torch.tensor(i).unsqueeze(0).cuda()
            #     _, pred_z_0, z_t = diffusion.p_losses(cldm, z_0, t, cond_aug)
            
            #     x_t = pure_cldm.vae_decode(z_t)
            #     pred_x_0 = pure_cldm.vae_decode(pred_z_0)        # b 3 448 448

            #     # JLP - crop only the text bbox for recognition
            #     cropped_imgs=[]
            #     for i in range(bs):
            #         img = pred_x_0[i]               # 3 448 448
            #         x1, y1, x2, y2 = boxes[i]      # x1 y1 x2 y2
            #         cropped_img = crop(img, y1, x1, y2-y1, x2-x1)   # format: x y w h
            #         resized_img = F.interpolate(cropped_img.unsqueeze(0), size=(32,128), mode='bilinear', align_corners=False)
            #         cropped_imgs.append(resized_img)
            #     cropped_imgs = torch.cat(cropped_imgs)

            #     rec_img = cropped_imgs.cuda()
            #     target_text = txt_encs.cuda()
            #     target_len = txt_lens.cuda()
            #     rec_feat = None
                
            #     rec_imgs = (rec_img, target_text, target_len, rec_feat)
            #     rec_outputs = recognizer(rec_imgs)
            #     rec_outputs = rec_outputs[0]

            #     print(rec_outputs[:,:4,:7])
            #     print('')
            #     print('='*50)
            #     # # JLP - see pred recognized texts 
            #     for i in range(bs):
            #         pred = rec_outputs[i]
            #         pred_idx = pred.argmax(dim=-1)
            #         # row_idx = torch.arange(pred.shape[0])
            #         # tmp2 = pred[row_idx, pred.argmax(dim=-1)]
            #         pred_word = decode(pred_idx)
            #         print('GT Word: ', text[i])
            #         print('Pred Word: ', pred_word)
            #         print('-'*50)

            #     loss_rec = rec_criterion(rec_outputs, target_text, target_len)
            #     print('OCR_WEIGHT: ', cfg.pho_args.ocr_loss_weight)
            #     print(f'loss_ocr_rec: {loss_rec.item():.3f}')
            #     print(f'loss_diff: {loss.item():.3f}')
            #     print('='*50)

            #     lq = (lq-lq.min())/(lq.max()-lq.min())
            #     x_t = (x_t-x_t.min())/(x_t.max()-x_t.min())
            #     pred_x_0 = (pred_x_0-pred_x_0.min())/(pred_x_0.max()-pred_x_0.min())

            #     # only label bbox for gt img
            #     gt = (gt-gt.min())/(gt.max()-gt.min())*255.0
            #     vis_gt = gt.squeeze().permute(1,2,0).detach().cpu().numpy().astype(np.uint8)  # 448 448 3
            #     vis_gt = vis_gt.copy()
            #     x1, y1, x2, y2 = boxes[i]
            #     x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
            #     cv2.rectangle(vis_gt, (x1,y1), (x2,y2), color=(0,255,0), thickness=4)
            #     cv2.imwrite(f'./img_gt.jpg', vis_gt[...,::-1])
            #     vis_gt = torch.tensor(vis_gt)
            #     vis_gt = vis_gt.cuda().permute(2,0,1).float() / 255.0
                
            #     rec_loss = str(round(loss_rec.item(),3))
            #     log_loss = f"{rec_loss.split('.')[0]}_{rec_loss.split('.')[-1]}"

            #     train_img_all = torch.cat([lq, x_t, pred_x_0, vis_gt.unsqueeze(0)], dim=-1)
            #     save_image(train_img_all, f'./vis/noisy_imgs/img{global_step}_t{t.item()}_gt{text[0]}_pred{pred_word}_recloss{log_loss}.jpg')
            #     print('TIMESTEP: ', t)

            pred_x_0 = pure_cldm.vae_decode(pred_z_0)        # b 3 448 448

            # JLP - crop only the text bbox for recognition
            cropped_imgs=[]
            for i in range(bs):
                img = pred_x_0[i]               # 3 448 448
                x1, y1, x2, y2 = boxes[i]      # x1 y1 x2 y2
                cropped_img = crop(img, y1, x1, y2-y1, x2-x1)   # format: x y w h
                resized_img = F.interpolate(cropped_img.unsqueeze(0), size=(32,128), mode='bilinear', align_corners=False)
                cropped_imgs.append(resized_img)
            cropped_imgs = torch.cat(cropped_imgs)

            rec_img = cropped_imgs.cuda()
            target_text = txt_encs.cuda()
            target_len = txt_lens.cuda()
            rec_feat = None
            
            # Recognizer model
            rec_imgs = (rec_img, target_text, target_len, rec_feat)
            rec_outputs = recognizer(rec_imgs)
            rec_outputs = rec_outputs[0]

            # # JLP - see pred recognized texts 
            print('')
            print('='*50)
            pred_words=[]
            for i in range(bs):
                pred = rec_outputs[i]
                pred_idx = pred.argmax(dim=-1)
                # row_idx = torch.arange(pred.shape[0])
                # tmp2 = pred[row_idx, pred.argmax(dim=-1)]
                pred_word = decode(pred_idx)
                pred_words.append(pred_word)
                print('GT Word: ', text[i])
                print('Pred Word: ', pred_words[i])
                print('-'*50)

            loss_rec = rec_criterion(rec_outputs, target_text, target_len)

            # TOTAL LOSS
            total_loss = loss + cfg.pho_args.ocr_loss_weight * loss_rec
            
            print('OCR_WEIGHT: ', cfg.pho_args.ocr_loss_weight)
            print(f'loss_ocr_rec: {loss_rec.item():.3f}')
            print(f'loss_ocr_rec_scaled: {cfg.pho_args.ocr_loss_weight * loss_rec.item():.3f}')
            print(f'loss_diff: {loss.item():.3f}')
            print('='*50)

            if global_step > 0:
                accelerator.backward(total_loss)
                opt.step()
                accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            # JLP - add additional loss
            step_ocr_rec_loss.append(loss_rec.item())
            # step_ocr_loss.append(ocr_loss.item())
            step_total_loss.append(total_loss.item())
            epoch_total_loss.append(total_loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log Training loss
            if global_step==1 or (global_step % cfg.train.log_train_every == 0 and global_step > 0):

                # Gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )

                avg_ocr_rec_loss = (
                    accelerator.gather(
                        torch.tensor(step_ocr_rec_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )

                avg_total_loss = (
                    accelerator.gather(
                        torch.tensor(step_total_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                step_ocr_rec_loss.clear()
                # step_ocr_loss.clear()
                step_total_loss.clear()
                
                if accelerator.is_main_process:
                    wandb.log({"loss/loss_simple_step": avg_loss}, commit=False)
                    wandb.log({'loss/ocr_rec_loss': avg_ocr_rec_loss}, commit=False)
                    # wandb.log({"loss/ocr_loss_simple_step": avg_ocr_loss}, commit=False)
                    wandb.log({"loss/total_loss_simple_step": avg_total_loss}, commit=False)
                    
            # Log Validation Loss
            if global_step==1 or (global_step % cfg.val.log_val_every == 0 and global_step > 0):
                print(f'Validation Loop') if accelerator.is_main_process else None
                total_val_loss = []
                for val_batch in val_loader:
                    to(val_batch, device)
                    val_batch = batch_transform(val_batch)
                    val_gt, val_lq, val_prompt, val_text, val_bbox, val_img_name  = val_batch
                    val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float()
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float()

                    with torch.no_grad():
                        val_z_0 = pure_cldm.vae_encode(val_gt)
                        val_clean = swinir(val_lq)
                        val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)
                        val_cond_aug = copy.deepcopy(val_cond)
                        if noise_aug_timestep > 0:
                            val_cond_aug["c_img"] = diffusion.q_sample(
                                x_start=val_cond_aug["c_img"],
                                t=torch.randint(
                                    0, noise_aug_timestep, (z_0.shape[0],), device=device
                                ),
                                noise=torch.randn_like(val_cond_aug["c_img"]),
                            )
                        val_t = torch.randint(
                            0, diffusion.num_timesteps, (val_z_0.shape[0],), device=device
                        )

                        val_loss, val_pred_z_0, _ = diffusion.p_losses(cldm, val_z_0, val_t, val_cond_aug)
                        total_val_loss.append(val_loss.item())
                        
                    accelerator.wait_for_everyone()
                    
                avg_val_loss = (
                    accelerator.gather(
                        torch.tensor(total_val_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                if accelerator.is_main_process:
                    wandb.log({"loss/val_loss_simple_step": avg_val_loss}, commit=False)
                
            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
            if global_step % cfg.train.save_unet == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = cldm.unet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}_unet.pt"
                    torch.save(checkpoint, ckpt_path)
                    
            # log train imgs
            if global_step==1 or (global_step % cfg.train.log_train_img_every == 0 and global_step > 0):
                cldm.eval()
                # log vis training imgs
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(bs, *z_0.shape[1:]),
                        cond=cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    if accelerator.is_main_process:
                        for i in range(bs):
                            vis_sample = pure_cldm.vae_decode(z)[i]
                            vis_gt = gt[i]              # 3 448 448
                            vis_lq = lq[i]              # 3 448 448
                            vis_clean = clean[i]        # 3 448 448
                            vis_pred_x0 = pred_x_0[i]   # 3 448 448

                            vis_gt = (vis_gt - vis_gt.min()) / (vis_gt.max()-vis_gt.min()) * 255.0
                            vis_lq = (vis_lq - vis_lq.min()) / (vis_lq.max() - vis_lq.min())
                            vis_clean  = (vis_clean  - vis_clean.min()) / (vis_clean.max() - vis_clean.min())
                            vis_pred_x0 = (vis_pred_x0 - vis_pred_x0.min()) / (vis_pred_x0.max() - vis_pred_x0.min())
                            vis_sample = (vis_sample - vis_sample.min()) / (vis_sample.max() - vis_sample.min())

                            # only label bbox for gt img
                            vis_gt = vis_gt.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)  # 448 448 3
                            vis_gt = vis_gt.copy()
                            x1, y1, x2, y2 = boxes[i]
                            x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                            cv2.rectangle(vis_gt, (x1,y1), (x2,y2), color=(0,255,0), thickness=4)

                            cv2.putText(vis_gt, str(t[i].item()), (350,380), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA)
                            cv2.putText(vis_gt, text[i], (350,400), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA)
                            cv2.putText(vis_gt, pred_words[i], (350,420), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA)
                            cv2.putText(vis_gt, str(round(loss_rec.item(),3)), (350,440), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA)
                            # cv2.imwrite(f'text.jpg', vis_gt[...,::-1])
                            vis_gt = torch.tensor(vis_gt)
                            vis_gt = vis_gt.cuda().permute(2,0,1).float() / 255.0

                            train_img_all = torch.cat([vis_lq, vis_clean, vis_pred_x0, vis_sample, vis_gt], dim=-1)
                            wandb.log({f'vis_train/train_img_all_{i}':wandb.Image(train_img_all, caption=f'lq_clean_predx0_sample_gt')})
                            # breakpoint()
                cldm.train()

            # log val imgs
            if global_step==1 or (global_step % cfg.val.log_val_img_every == 0 and global_step > 0):
                cldm.eval()
                with torch.no_grad():
                    # Log Validation Images
                    M = cfg.val.log_val_num_img
                    val_log_clean = val_clean[:M]
                    val_log_cond = {k: v[:M] for k, v in val_cond.items()}
                    val_log_cond_aug = {k: v[:M] for k, v in val_cond_aug.items()}
                    val_log_gt, val_log_lq = val_gt[:M], val_lq[:M]
                    val_log_prompt = val_prompt[:M]
                    val_z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(val_log_gt), *val_z_0.shape[1:]),
                        cond=val_log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )

                    if accelerator.is_main_process:
                        for i in range(M):
                            vis_sample = pure_cldm.vae_decode(val_z)[i]
                            vis_pred_x0 = pure_cldm.vae_decode(val_pred_z_0)[i]
                            vis_gt = val_log_gt[i]
                            vis_lq = val_log_lq[i]
                            vis_clean = val_log_clean[i]

                            vis_sample = (vis_sample - vis_sample.min()) / (vis_sample.max() - vis_sample.min())
                            vis_pred_x0 = (vis_pred_x0 - vis_pred_x0.min()) / (vis_pred_x0.max() - vis_pred_x0.min())
                            vis_gt = (vis_gt - vis_gt.min()) / (vis_gt.max() - vis_gt.min())
                            vis_lq = (vis_lq - vis_lq.min()) / (vis_lq.max()-vis_lq.min())
                            vis_clean = (vis_clean - vis_clean.min()) / (vis_clean.max() - vis_clean.min())

                            val_img_all = torch.cat([vis_lq, vis_clean, vis_pred_x0, vis_sample, vis_gt], dim=-1)
                            wandb.log({f'vis_val/val_img_all_{i}':wandb.Image(val_img_all, caption=f'lq_clean_predx0_sample_gt')})

                cldm.train()
            
            if accelerator.is_main_process:
                wandb.log({"step": global_step}, commit=True)
            
            accelerator.wait_for_everyone()
            
            if global_step == max_steps:
                break
            torch.cuda.empty_cache()
        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_total_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_total_loss.clear()
        if accelerator.is_main_process:
            wandb.log({'epoch': epoch})
            wandb.log({"loss/loss_simple_epoch": avg_epoch_loss}, step=global_step)

    if accelerator.is_main_process:
        print("done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bridge_config", default="", metavar="FILE", help="path to config file")
    args = parser.parse_args()
    main(args)
    # scp -r checkpoint-9.pth cvlab08@163.152.163.126:/home/cvlab08/projects/data/jinlovespho/nips25/github/diffbir_finetuning/pretrained_weights
    # scp -r checkpoint-9.pth cvlab12@163.152.163.141:/media/dataset2/jinlovespho/DiffBIR/pretrained_weights
