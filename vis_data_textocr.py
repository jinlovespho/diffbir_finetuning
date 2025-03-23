import cv2 
from PIL import Image 
import numpy as np 
import os 
import json

data_path = './generated_data/TextOCR/train'
ann_path = './generated_data/ocr/train_dataset_modified_filtered.json'

img_ids = sorted(os.listdir(data_path))
img_paths = [f'{data_path}/{img_id}' for img_id in img_ids]

with open(ann_path, 'r') as file:
    anns = json.load(file)

# img0 = np.array(Image.open(f'{data_path}/4a96075fc4d92d1b_crop_0.jpg'))     # 512 512 3
# cv2.imwrite(f'./tmp3.jpg', img0[:,:,::-1])
# ann0 = anns['4a96075fc4d92d1b']['0']
# for i in range(len(ann0)):
#     box = ann0[i]['bbox']
#     text = ann0[i]['text']
#     # print('Text: ', text)
#     if text == '.':
#         continue
#     a, b, c, d = map(int, box)  # XYWH format
#     cv2.rectangle(img0, (a,b), (a+c,b+d), (0,255,0), 2)
#     cv2.putText(img0, text, (a,b), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
# cv2.imwrite(f'./tmp4.jpg', img0[:,:,::-1])
# breakpoint()

for j in range(20):
    
    img_path = img_paths[j]
    img = np.array(Image.open(img_path))
    img_id = img_path.split('/')[-1].split('_')[0]
    crop_id = img_path.split('/')[-1].split('_')[-1].split('.')[0]
    img_ann = anns[img_id][crop_id]
    num_crops = len(img_ann)

    # cv2.imwrite(f'./vis.jpg', img[:,:,::-1])
    for i in range(num_crops):
        box = img_ann[i]['bbox']
        text = img_ann[i]['text']
        # print('Text: ', text)
        # if text == '.':
        #     continue
        a, b, c, d = map(int, box)  # XYWH format
        cv2.rectangle(img, (a,b), (a+c,b+d), (0,255,0), 2)
        cv2.putText(img, text, (a,b), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

    cv2.imwrite(f'./vis/examples_of_train_data/textocr_{img_id}_crop_{crop_id}.jpg', img[:,:,::-1])
    # cv2.imwrite(f'./vis.jpg', img[:,:,::-1])
    print(img_path)
    print(img_id, crop_id, img_ann)
breakpoint()
