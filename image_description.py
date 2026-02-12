import argparse
import os
import random
import json
import sys
import time
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast

# 强制使用正确的MiniGPT-4路径
sys.path.insert(0, "/root/MiniGPT-4")

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# 手动定义CONV_VISION
from minigpt4.conversation.conversation import Chat, Conversation, SeparatorStyle

# 手动定义CONV_VISION
CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

# 确保导入注册模块
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Image Description Generator")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image directory")
    parser.add_argument("--json_file", type=str, required=True, help="Path to JSON annotation file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images to process in one batch")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file."
    )
    args = parser.parse_args()
    return args


def setup_seeds():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def enhance_prompt(chat, base_prompt, chat_state, img_list, max_tokens=60, use_amp=False):
    """增强基础提示词，生成更丰富的描述，但限制在指定词数内"""
    instruction = (
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. "
        "Evaluate the level of detail in the user prompt:\n"
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n"
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n"
        f"IMPORTANT: Your enhanced description MUST be under {max_tokens} tokens (approximately 50-60 words). "
        "Focus on the most essential visual details only. Be concise but descriptive.\n"
        "Examples of enhancements:\n"
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n"
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\n"
        f"User Prompt: {base_prompt}\nEnhanced description:"
    )

    chat.ask(instruction, chat_state)

    if use_amp:
        with autocast():
            enhanced_prompt = chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0
            )[0]
    else:
        enhanced_prompt = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1.0
        )[0]

    # 处理返回的结果，移除可能的前缀
    if "Enhanced description:" in enhanced_prompt:
        enhanced_prompt = enhanced_prompt.split("Enhanced description:")[1].strip()

    # 确保不超过60个单词
    words = enhanced_prompt.split()
    if len(words) > 60:
        enhanced_prompt = " ".join(words[:60])
        enhanced_prompt += "..."

    return enhanced_prompt


def generate_concise_description(chat, image_info, chat_state, img_list, use_amp=False):
    """生成类似示例的简洁描述"""
    # 生成背景提示词
    background_prompt = "Describe the background or environment in this image in one brief sentence."
    chat.ask(background_prompt, chat_state)

    if use_amp:
        with autocast():
            background_description = chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0
            )[0].split('.')[0]
    else:
        background_description = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1.0
        )[0].split('.')[0]

    # 基础提示词
    base_prompt = f"An image showing {', '.join([a['category_name'] for a in image_info['annotations']])}"

    # 为每个区域生成简洁描述
    regional_descriptions = {}
    for i, annotation in enumerate(image_info["annotations"]):
        category_name = annotation["category_name"]
        mask = annotation["mask"]

        # 获取简洁的区域描述
        region_prompt = f"Describe the {category_name} in the image in one concise sentence."
        chat.ask(region_prompt, chat_state)

        if use_amp:
            with autocast():
                region_description = chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    num_beams=1,
                    temperature=1.0
                )[0].split('.')[0]
        else:
            region_description = chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0
            )[0].split('.')[0]

        # 添加到区域描述中
        width = image_info["image"]["width"]
        height = image_info["image"]["height"]
        regional_descriptions[str(i)] = {
            "description": region_description,
            "mask": [
                f"int({mask[0] / width:.2f} * image_width)",
                f"int({mask[1] / height:.2f} * image_height)",
                f"int({mask[2] / width:.2f} * image_width)",
                f"int({mask[3] / height:.2f} * image_height)"
            ]
        }

    # 创建并返回简洁描述
    concise_description = {
        "base_prompt": base_prompt,
        "background_prompt": background_description,
        "regional_prompt_mask_pairs": regional_descriptions
    }

    return concise_description


def print_concise_prompt(concise_description):
    """打印简洁的描述提示"""
    print("\n# 简洁描述提示")
    print("# 基础提示词和背景提示词")
    print(f"base_prompt = \"{concise_description['base_prompt']}\"")
    print(f"background_prompt = \"{concise_description['background_prompt']}\"")

    print("\n# 区域提示词和蒙版")
    print("regional_prompt_mask_pairs = {")

    for region_id, region_info in concise_description["regional_prompt_mask_pairs"].items():
        # 添加区域序号和描述
        print(f"    \"{region_id}\": {{")
        print(f"        \"description\": \"{region_info['description']}\",")
        print(f"        \"mask\": [{region_info['mask'][0]}, {region_info['mask'][1]},")
        print(f"                 {region_info['mask'][2]}, {region_info['mask'][3]}]")
        print(f"    }}{',' if int(region_id) < len(concise_description['regional_prompt_mask_pairs']) - 1 else ''}")

    print("}")


def process_image(image_path, image_info, chat, output_file=None, use_amp=False):
    """处理单个图像并生成描述"""
    # 初始化对话
    chat_state = CONV_VISION.copy()
    img_list = []

    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        print(f"Processing image: {image_path}")

        # 上传图像
        chat.upload_img(image, chat_state, img_list)

        # 生成整体图像描述 - 使用混合精度
        overall_prompt = "Please describe what you see in this image in detail."
        chat.ask(overall_prompt, chat_state)

        if use_amp:
            with autocast():
                overall_response = chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    num_beams=1,
                    temperature=1.0
                )[0]
        else:
            overall_response = chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0
            )[0]

        # 生成背景描述 - 使用混合精度
        background_prompt = "Describe the background or environment in this image."
        chat.ask(background_prompt, chat_state)

        if use_amp:
            with autocast():
                background_response = chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    num_beams=1,
                    temperature=1.0
                )[0]
        else:
            background_response = chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0
            )[0]

        # 构建基础提示词
        base_prompt = f"An image showing {', '.join([a['category_name'] for a in image_info['annotations']])}"

        # 创建增强的提示词
        enhanced_prompt = enhance_prompt(chat, base_prompt, chat_state, img_list, use_amp=use_amp)

        # 生成简洁描述
        concise_description = generate_concise_description(chat, image_info, chat_state, img_list, use_amp=use_amp)

        # 打印简洁描述提示
        print_concise_prompt(concise_description)

        # 格式化输出
        results = {
            "image_path": image_path,
            "overall_description": overall_response,
            "background_description": background_response,
            "regional_descriptions": [],
            "formatted_output": {
                "base_prompt": base_prompt,
                "enhanced_prompt": enhanced_prompt,
                "background_prompt": background_response.split('.')[0],
                "regional_prompt_mask_pairs": {}
            },
            "concise_description": concise_description
        }

        # 为每个注释生成区域描述
        for i, annotation in enumerate(image_info["annotations"]):
            category_name = annotation["category_name"]
            bbox = annotation["bbox"]
            mask = annotation["mask"]

            region_prompt = f"Focus on the {category_name} in the image with bounding box coordinates {bbox}. Describe it in detail."
            chat.ask(region_prompt, chat_state)

            if use_amp:
                with autocast():
                    region_response = chat.answer(
                        conv=chat_state,
                        img_list=img_list,
                        num_beams=1,
                        temperature=1.0
                    )[0]
            else:
                region_response = chat.answer(
                    conv=chat_state,
                    img_list=img_list,
                    num_beams=1,
                    temperature=1.0
                )[0]

            # 添加到结果中
            results["regional_descriptions"].append({
                "region_id": i,
                "category_name": category_name,
                "bbox": bbox,
                "mask": mask,
                "description": region_response
            })

            # 确保annotation中有description字段
            if "description" not in annotation:
                description_prompt = f"Give a brief description of the {annotation['category_name']} in one sentence."
                chat.ask(description_prompt, chat_state)

                if use_amp:
                    with autocast():
                        description = chat.answer(
                            conv=chat_state,
                            img_list=img_list,
                            num_beams=1,
                            temperature=1.0
                        )[0]
                else:
                    description = chat.answer(
                        conv=chat_state,
                        img_list=img_list,
                        num_beams=1,
                        temperature=1.0
                    )[0]

                annotation["description"] = description

            # 为每个区域创建增强的提示词
            region_base_prompt = f"A {category_name} in an image"
            enhanced_region_prompt = enhance_prompt(chat, region_base_prompt, chat_state, img_list, use_amp=use_amp)

            # 将区域信息添加到格式化输出中
            width = image_info["image"]["width"]
            height = image_info["image"]["height"]
            results["formatted_output"]["regional_prompt_mask_pairs"][str(i)] = {
                "description": annotation.get("description", f"A {category_name}"),
                "enhanced_description": enhanced_region_prompt,
                "mask": [
                    f"int({mask[0] / width:.4f} * image_width)",
                    f"int({mask[1] / height:.4f} * image_height)",
                    f"int({mask[2] / width:.4f} * image_width)",
                    f"int({mask[3] / height:.4f} * image_height)"
                ]
            }

        # 如果指定了输出文件，保存结果
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                print(f"Results saved to {output_file}")

        # 清理GPU内存
        torch.cuda.empty_cache()

        return image_path

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # 清理GPU内存
        torch.cuda.empty_cache()
        return None


def batch_process_images(chat, image_paths, annotations_data, output_dir, batch_size=4, use_amp=False):
    """按批次顺序处理图像，以提高GPU利用率但避免CUDA错误"""
    results = []
    total = len(image_paths)

    # 按批次处理图像
    for i in range(0, total, batch_size):
        batch = image_paths[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ({len(batch)} images)")

        for image_path in batch:
            image_filename = os.path.basename(image_path)
            if image_filename in annotations_data:
                output_file = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_description.json")
                result = process_image(image_path, annotations_data[image_filename], chat, output_file, use_amp)
                if result:
                    results.append(result)

    return results


def main():
    print('Initializing Stable Image Description Generator')
    args = parse_args()

    import minigpt4
    print(f"Using MiniGPT-4 from: {os.path.dirname(minigpt4.__file__)}")

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载JSON注释
    try:
        with open(args.json_file, 'r') as f:
            annotations_data = json.load(f)
        print(f"Successfully loaded JSON annotations: {args.json_file}")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        exit(1)

    # 设置随机种子
    setup_seeds()

    # 初始化配置
    cfg = Config(args)

    # 初始化模型
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()

    # 初始化视觉处理器
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}')
    print('Model Initialization Finished')

    # 收集要处理的图像
    image_paths = []
    if os.path.isfile(args.image_path):
        # 单个文件处理
        image_filename = os.path.basename(args.image_path)
        if image_filename in annotations_data:
            image_paths.append(args.image_path)
        else:
            print(f"Image {image_filename} not found in annotations")
    else:
        # 目录处理
        for root, _, files in os.walk(args.image_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    if file in annotations_data:
                        image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("No matching images found!")
        exit(1)

    print(f"Found {len(image_paths)} images to process")

    # 按批次处理图像
    start_time = time.time()
    results = batch_process_images(
        chat,
        image_paths,
        annotations_data,
        args.output_dir,
        batch_size=args.batch_size,
        use_amp=args.use_amp
    )
    end_time = time.time()

    print(f"Successfully processed {len(results)} images in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()