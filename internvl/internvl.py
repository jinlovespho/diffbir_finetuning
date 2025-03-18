import math
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from .conversation import get_conv_template
import torch.nn.functional as F
from copy import deepcopy
import transformers
from typing import Dict
import numpy as np
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from .constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)

import json
from PIL import Image, ImageDraw
import os

def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False
        
def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

device_map = split_model('InternVL2_5-4B')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5)
    ])
    return transform

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_input_ids(model, tokenizer, question, generation_config, history=None,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False):

    if history is None and pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    history = [] if history is None else history
    for (old_question, old_answer) in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model.device)
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    generation_config['eos_token_id'] = eos_token_id
    generation_config['return_dict'] = True
    return input_ids, generation_config
 
def preprocess_internvl2_5(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,  IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        num_image: int = 1
) -> Dict:
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message
        # system_prompt = None

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}'
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('gpt')
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids
    
    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'human':
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == 'gpt':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    ) 

def get_ret(model, tokenizer, data_item, max_num=12):
    breakpoint()
    # Ensure the first conversation contains an image placeholder
    if '<image>' not in data_item['conversations'][0]['value']:
        data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']   # 이미지를 user prompt 앞쪽에 append하는 것
    pixel_values = data_item['image']
    
    # Ensure that there is only one patch if dynamic image size is not enabled
    num_patches = pixel_values.size(0)
    #assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
    
    ret = preprocess_internvl2_5(model.template, [deepcopy(data_item['conversations'])],
                                tokenizer, [model.num_image_token * num_patches],
                                group_by_length=False,
                                use_packed_ds=False, ds_name=None)

    # Calculate position_ids for packed dataset
    position_ids = ret['attention_mask'].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
    image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    
    assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is None'

    # Create the final return dictionary
    ret = dict(
        input_ids=ret['input_ids'].to(model.device),
        labels=ret['labels'].to(model.device),
        attention_mask=ret['attention_mask'].to(model.device),
        position_ids=position_ids.to(model.device),
        pixel_values=pixel_values.to(model.device).to(torch.bfloat16),
        image_flags=torch.tensor([1] * num_patches, dtype=torch.long).to(model.device)
    )
    return ret
           
def load_internvl():
    model = 'InternVL2_5-1B'
    path = f"OpenGVLab/{model}"

    device_map = split_model(model)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map)
    
    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    model.language_model._set_gradient_checkpointing()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    return model, tokenizer

def get_data_item(decoded_pred_z0, bbox, text):
    data_item = {
        "image": decoded_pred_z0,
        "conversations": [
            {
                "from": "human",
                "value": f"""
                        Please perform OCR on the image within the following bounding box coordinates: [x, y, w, h].
                        <box> [{[coord.item() for coord in bbox]}] </box>
                        
                        The expected output format is as follows:
                        - Provide the text extracted from the image within the specified bounding box.
                        - The result should be plain text with no additional formatting or explanations.
                        - If text is unreadable, please provide an empty string ("").
                        
                        Please return only the extracted text, with no extra information or context.
                        
                        Example Output:
                        - 'Hello, World!'
                        - '' (Case of unreadable text)
                        """
            },
            {
                "from": "gpt",
                "value": text
            }
        ]
    }
    
    return data_item


def get_model_chat(model, tokenizer, data_item):
    model.eval()
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    pixel_values = data_item['image'].to(torch.bfloat16)
    response = model.chat(tokenizer, pixel_values, data_item['conversations'][0]['value'], generation_config,)
    model.train()
    
    return response

def resize_bbox(bbox, orig_w, orig_h, new_w, new_h):
    x, y, w, h = bbox
    scale_w, scale_h = new_w / orig_w, new_h / orig_h
    return (x * scale_w, y * scale_h, w * scale_w, h * scale_h)

def crop_tensor_image(img_tensor, bbox, input_size=448):
    #bbox = [coord.item() for coord in bbox]
    bbox = resize_bbox(bbox, 512, 512, input_size, input_size)
    
    cropped_imgs=[]
    new_bboxes = []
    for i in range(len(bbox[0])):
        x,y,w,h = int(bbox[0][i].item()), int(bbox[1][i].item()), int(bbox[2][i].item()), int(bbox[3][i].item())
        cropped = T.functional.crop(img_tensor[i],  y, x, h, w)
        resized = T.functional.resize(cropped, (input_size,input_size), antialias=True) 
        cropped_imgs.append(resized)

        new_bbox = [x,y,w,h]
        new_bboxes.append(new_bbox)

    cropped_img = torch.stack(cropped_imgs)
    return cropped_img, new_bboxes

    for val in bbox:
        x,y,w,h = map(int, val.tolist())
        

    x,y,w,h = map(int, bbox)
    cropped = T.functional.crop(img_tensor[0],  y, x, h, w)
    resized = T.functional.resize(cropped, (input_size,input_size), antialias=True) 
    
    return resized.unsqueeze(0)

### Test the model ###
def crop_and_load_image(image_path, bbox, input_size=448, max_num=12):
    image = Image.open(image_path)
    # x, y, w, h to x1, y1, x2, y2
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    image = image.crop(bbox)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).to(torch.float32)
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.cpu().numpy()
    tensor = (tensor * 0.5) + 0.5
    tensor = np.clip(tensor, 0, 1)
    tensor = (tensor * 255).astype(np.uint8)
    
    return Image.fromarray(tensor)

def display_bbox(img, bbox):
    # bbox : [x, y, w, h]
    img = img.resize((512, 512))
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline='red', width=3)
    

    return img

def logits_to_string(labels, logits, tokenizer):
    predicted_tokens = logits.argmax(dim=-1)  # shape: [1, 16384]
    mask = labels != IGNORE_TOKEN_ID  # shape: [1, 16384]
    mask = mask[:, 1:]  # shape: [1, 16383]
    labels = labels[:, :-1,:]
    response = tokenizer.batch_decode(masked_predicted_tokens, skip_special_tokens=True)[0]
    return response

def slice_unknown_vocab(labels_ids, predictions):
    selected = labels_ids != -100
    prediction_selected = np.concatenate([selected[:, 1:], np.zeros((selected.shape[0], 1), dtype=bool)], axis=1)

    new_labels_ids, new_predictions = [], []
    for i, (label_id, prediction) in enumerate(zip(labels_ids, predictions)):
        new_labels_ids.append(label_id[selected[i]])
        new_predictions.append(prediction[prediction_selected[i]])

    return new_labels_ids, new_predictions

def decode_predictions(tokenizer, logits, labels):
    prediction_ids = logits.argmax(axis=-1)
    labels_ids, prediction_ids = slice_unknown_vocab(labels, prediction_ids)
    label_text = tokenizer.batch_decode(labels_ids)
    prediction_text = tokenizer.batch_decode(prediction_ids)

    return {"labels": label_text, "predictions": prediction_text}

def test():
    model, tokenizer = load_internvl()
    mode = 'image'
    img_path_list = [f'/media/dataset2/jaewon/generated_data/lsdir/train/{mode}/512/0003289_1.png',
                    f'/media/dataset2/jaewon/generated_data/lsdir/train/{mode}/512/0003185_1.png',
                    f'/media/dataset2/jaewon/generated_data/lsdir/train/{mode}/512/0003084_1.png',
                    f'/media/dataset2/jaewon/generated_data/lsdir/train/{mode}/512/0003293_1.png',]
    json_path = '/home/cvlab12/project/jaewon/text_restoration/data_gen/ocr/lsdir_adjust_label_train.json'
    json_file = json.load(open(json_path, 'r'))
    
    output_dir = 'vlm_test/prompt_1_lr_image'
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in img_path_list:
        img_id, idx = img_path.split('/')[-1].split('.')[0].split('_')
        
        text = json_file[img_id][idx]['text']
        bbox = json_file[img_id][idx]['bbox']
        
        pixel_values1 = load_image(img_path, max_num=1).to(torch.bfloat16).cuda()
        pixel_values2 = crop_and_load_image(img_path, bbox, max_num=1).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        
        bbox = [torch.tensor(coord) for coord in bbox]
        data_item = get_data_item(pixel_values, bbox, text)
        scene_data_item = get_data_item(pixel_values1, bbox, text)
        cropped_data_item = get_data_item(pixel_values2, bbox, text)
        
        img = tensor_to_image(pixel_values1)
        img = display_bbox(img, bbox)
        img.save(f'{output_dir}/{img_id}_bbox.png')
        croppped_img = tensor_to_image(pixel_values2)
        croppped_img.save(f'{output_dir}/{img_id}_cropped.png')
        
        response_list = []
        whole_response_list = []
        cropped_response_list = []
        
        loss_list = []
        whole_loss_list = []
        cropped_loss_list = []
        
        print(f'GT: ', text)
        # Set seed for reproducibility
        for i,item in enumerate([data_item, scene_data_item, cropped_data_item]):
            ret = get_ret(model, tokenizer, item)
            with open(f'{output_dir}/input.txt', 'w') as f:
                f.write(f'{item["conversations"][0]["value"]}\n')
            
            for _ in range(5):
                torch.manual_seed(_)
                with torch.no_grad():
                    output = model(**ret)
                    ocr_loss = output.loss
                    logits = output.logits
                    
                    # pred_dict = decode_predictions(tokenizer, logits.cpu().numpy(), ret['labels'].cpu().numpy())
                    # response = pred_dict['predictions'][0]
                torch.manual_seed(_)
                response = get_model_chat(model, tokenizer, item)

                if i==0:
                    response_list.append(response)
                    loss_list.append(ocr_loss.item())
                elif i==1:
                    whole_response_list.append(response)
                    whole_loss_list.append(ocr_loss.item())
                elif i==2:
                    cropped_response_list.append(response)
                    cropped_loss_list.append(ocr_loss.item())
                
                torch.cuda.empty_cache()
            del ret
        # Save the responses as txt files
        with open(f'{output_dir}/{img_id}_response.txt', 'w') as f:
            f.write(f'GT: {text}\n')
            f.write(f'Concatenated Image\n')
            for i, response in enumerate(response_list):
                f.write(f'[{i}] {response}\n')
            f.write(f'Whole Image\n')
            for i, response in enumerate(whole_response_list):
                f.write(f'[{i}] {response}\n')
            f.write(f'Cropped Image\n')
            for i, response in enumerate(cropped_response_list):
                f.write(f'[{i}] {response}\n')
            
        # Save the losses as txt files
        with open(f'{output_dir}/{img_id}_loss.txt', 'w') as f:
            f.write(f'Concatenated Image\n')
            for i, loss in enumerate(loss_list):
                f.write(f'[{i}] {loss}\n')
            f.write(f'Whole Image\n')
            for i, loss in enumerate(whole_loss_list):
                f.write(f'[{i}] {loss}\n')
            f.write(f'Cropped Image\n')
            for i, loss in enumerate(cropped_loss_list):
                f.write(f'[{i}] {loss}\n')