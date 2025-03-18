from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)
from .utils_ocr_vlm_filter import load_pair_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config

import cv2 
import string

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


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
            s.append(voc.index(rec[i]))
        elif i == len(rec):
            s.append(94)
        else:
            s.append(95)
    return s

class CodeformerDataset(data.Dataset):

    def __init__(
        self,
        file_list: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        mode = 'train',
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        # breakpoint()
        self.file_list = file_list
        self.image_files = load_pair_list(file_list, mode)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

        # # JLP - vis training dataset image, bbox, and text
        # for i in range(15):
        #     file = self.image_files[i]
        #     dataset_name = file['image_path'].split('/')[6]
        #     img_name = file['img_name']
        #     img_path = file['image_path']
        #     txt = file['text']
        #     box = file['bbox']
        #     x,y,w,h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        #     img = cv2.imread(img_path)  # h w 3
        #     cv2.rectangle(img, (x,y), (x+w,y+h) , color= (0,255,0), thickness=2)
        #     cv2.imwrite(f'./vis/{dataset_name}_{img_name}_{txt}.jpg', img)

    
    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        img_gt = None
        while img_gt is None:
            # breakpoint()
            # load meta file
            image_file = self.image_files[index]
            gt_path = image_file["image_path"]
            lq_path = image_file["lr_image_path"]
            prompt = image_file["prompt"]
            text = image_file["text"]
            bbox = image_file["bbox"]
            img_name = image_file['img_name']
            img_gt = self.load_gt_image(gt_path)
            img_lq = self.load_gt_image(lq_path)
            if img_gt is None:
                print(f"filed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        h, w, _ = img_gt.shape
        if np.random.uniform() < 0.5:
            prompt = ""

        # BGR to RGB, [-1, 1]
        gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        lq = img_lq[..., ::-1].astype(np.float32)

        # JLP - added ocr tokenizer which is pretty much just char level encoding
        # text_enc = encode(text)

        return gt, lq, prompt, text, bbox, img_name

    def __len__(self) -> int:
        return len(self.image_files)
