from datasets import Dataset, DatasetDict, Features, Image, Value, Sequence
import os
from datasets import concatenate_datasets
import json
import cv2

def load_lsdir(path):
    image_folder = f"{path}/lsdir/train/gt/512"
    condition_folder = f"{path}/lsdir/train/image/512"
    adjusted_bbox = json.load(open(f"{path}/ocr/lsdir_adjust_label_train.json"))
    image_files = sorted(os.listdir(image_folder))
    #condition_files = sorted(os.listdir(condition_folder))

    prompts = ["" for _ in range(len(image_files))]

    data = []
    for img, prompt in zip(image_files, prompts):
        image_path = os.path.join(image_folder, img)
        condition_path = os.path.join(condition_folder, img)
        img_name , img_id = img.split('.')[0].split('_')
        bbox = adjusted_bbox[img_name][img_id]['bbox'] 
        data.append({
            "image": image_path,
            "conditioning_image": condition_path,
            "text": prompt,
            'bbox': bbox
        })

    features = Features({
        "image": Image(),
        "conditioning_image": Image(),
        "text": Value("string"),
        "bbox": Sequence(Value("float"))
    })

    dataset = Dataset.from_list(data, features=features)
    return dataset

def load_realsr(path):
    image_folder = f"{path}/realsr/train/gt/512"
    condition_folder = f"{path}/realsr/train/image/512"
    adjusted_bbox = json.load(open(f"{path}/ocr/realsr_adjust_label_train.json"))
    image_files = sorted(os.listdir(image_folder))
    #condition_files = sorted(os.listdir(condition_folder))

    prompts = ["" for _ in range(len(image_files))]

    data = []
    for img, prompt in zip(image_files, prompts):
        image_path = os.path.join(image_folder, img)
        condition_path = os.path.join(condition_folder, img)
        img_name = '_'.join(img.split('.')[0].split('_')[0:2])
        img_id = img.split('.')[0].split('_')[2]
        bbox = adjusted_bbox[img_name][img_id]['bbox'] 
        data.append({
            "image": image_path,
            "conditioning_image": condition_path,
            "text": prompt,
            'bbox': bbox
        })

    features = Features({
        "image": Image(),
        "conditioning_image": Image(),
        "text": Value("string"),
        "bbox": Sequence(Value("float"))
    })

    dataset = Dataset.from_list(data, features=features)
    return dataset

def load_drealsr(path):
    image_folder = f"{path}/drealsr/train/gt/512"
    condition_folder = f"{path}/drealsr/train/image/512"
    adjusted_bbox = json.load(open(f"{path}/ocr/drealsr_adjust_label_train.json"))
    image_files = sorted(os.listdir(image_folder))
    #condition_files = sorted(os.listdir(condition_folder))

    prompts = ["" for _ in range(len(image_files))]

    data = []
    for img, prompt in zip(image_files, prompts):
        image_path = os.path.join(image_folder, img)
        condition_path = os.path.join(condition_folder, img)
        img_name = '_'.join(img.split('.')[0].split('_')[:len(img.split('.')[0].split('_'))-1])
        img_id = img.split('.')[0].split('_')[len(img.split('.')[0].split('_'))-1]
        bbox = adjusted_bbox[img_name][img_id]['bbox'] 
        data.append({
            "image": image_path,
            "conditioning_image": condition_path,
            "text": prompt,
            'bbox': bbox
        })

    features = Features({
        "image": Image(),
        "conditioning_image": Image(),
        "text": Value("string"),
        "bbox": Sequence(Value("float"))
    })

    dataset = Dataset.from_list(data, features=features)
    return dataset 

def load_textocr(path, split):
    image_folder = f"{path}/TextOCR/{split}"
    anns = json.load(open(f"{path}/ocr/{split}_dataset_modified_filtered.json"))
    image_files = sorted(os.listdir(image_folder))

    data = []
    for img in image_files:
        image_path = os.path.join(image_folder, img)
        # condition_path = os.path.join(condition_folder, img)
        img_id = image_path.split('/')[-1].split('_')[0]
        crop_id = image_path.split('/')[-1].split('_')[-1].split('.')[0]
        img_ann = anns[img_id][crop_id]
        num_crops = len(img_ann)

        boxes=[]
        texts=[]
        prompts=[]
        for i in range(num_crops):
            box = img_ann[i]['bbox']
            text = img_ann[i]['text']
            # print('Text: ', text)
            if text == '.':
                continue
            a, b, c, d = map(int, box)  # XYWH format
            box=[a,b,c,d]
            boxes.append(box)
            texts.append(text)
            prompts.append(f'A high-quality photo of the word {text}')
            # cv2.rectangle(img, (a,b), (a+c,b+d), (0,255,0), 2)
            # cv2.putText(img, text, (a,b), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

        data.append({
            "image": image_path,
            "text": texts,
            'bbox': boxes,
            'prompt':prompts
        })

    features = Features({
        "image": Image(),
        "text": Sequence(Value("string")),  # List of strings
        "bbox": Sequence(Sequence(Value("int32"))),  # List of lists of integers
        "prompt": Sequence(Value("string")),  # List of strings
    })

    dataset = Dataset.from_list(data, features=features)
    return dataset


def load_custom_dataset(path, split='train'):
    # lsdir = load_lsdir(path)
    # realsr = load_realsr(path)
    # drealsr = load_drealsr(path)
    textocr = load_textocr(path, split)
    dataset = concatenate_datasets([textocr])

    if split == 'train':
        dataset = DatasetDict({
            "train": dataset
        })

    elif split == 'val':
        dataset = DatasetDict({
            "val": dataset
        })

    return dataset


if __name__ == '__main__':
    path='/media/dataset2/jaewon/generated_data'
    dataset = load_custom_dataset(path=path, split='train')
    breakpoint()