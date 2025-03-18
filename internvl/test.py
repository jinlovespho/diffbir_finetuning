import math
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from conversation import get_conv_template
import torch.nn.functional as F
from copy import deepcopy
import transformers
from typing import Dict
import numpy as np
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)

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
        T.Normalize(mean=MEAN, std=STD)
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
    # Build transformation function
    transform = build_transform(448)

    # Ensure the first conversation contains an image placeholder
    if '<image>' not in data_item['conversations'][0]['value']:
        data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

    # Load the image using tcs_loader if available, otherwise use PIL
    pixel_values = load_image(data_item['image'], max_num=max_num)
    
    # Ensure that there is only one patch if dynamic image size is not enabled
    num_patches = pixel_values.size(0)
    
    #if not self.dynamic_image_size:
    #    assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

    # Preprocess the conversations and generate the return dictionary
    # Orgin Code:
    # ret = preprocess_internvl2_5(model.template, [deepcopy(data_item['conversations'])],
    #                             tokenizer, [model.num_image_token * num_patches],
    #                             group_by_length=model.group_by_length,
    #                             use_packed_ds=model.use_packed_ds, ds_name=model.ds_name)
    
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
           
# pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
# question = '<image>\nCan you pull all textual information from the images?'
# generation_config = dict(max_new_tokens=1024, do_sample=True)

model = 'InternVL2_5-1B'
path = f"OpenGVLab/{model}"

# Prepare the image
img_path = '../InternVL_org/OCR_eval_data/scene_text_rec/IMG_2257.JPG'
data_item = {
    "image": img_path,
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nOCR this image:"
        },
        {
            "from": "gpt",
            "value": "广告招商 4000 10 4008 北京天象世纪园際康告有限公司 www.86ad.com"
        }
    ]
}
device_map = split_model(model)
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map)

# _freeze_params(model.vision_model)
# _freeze_params(model.language_model)
# _freeze_params(model.mlp1)

model.vision_model.gradient_checkpointing = True
model.vision_model.encoder.gradient_checkpointing = True
model.language_model._set_gradient_checkpointing()
        
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
model.img_context_token_id = img_context_token_id

ret = get_ret(model, tokenizer, data_item)

response = model(**ret)
breakpoint()


