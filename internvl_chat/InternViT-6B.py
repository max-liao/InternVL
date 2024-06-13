from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

# Mean and standard deviation values for normalization, using ImageNet dataset statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Function to build image transformation pipeline
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            # Ensure image is in RGB mode
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            # Resize the image
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            # Convert image to tensor
            T.ToTensor(),
            # Normalize the image
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform

# Function to find the closest aspect ratio
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if (ratio_diff < best_ratio_diff):
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif (ratio_diff == best_ratio_diff):
            if (area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
                best_ratio = ratio
    return best_ratio

# Function to dynamically preprocess the image
def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate target aspect ratios
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if (i * j <= max_num and i * j >= min_num)
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest target aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        # Define box to crop image into blocks
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # Crop the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# Function to load and preprocess an image
def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Define the function to process images and ask a question
def process_images_and_ask_question(imagePaths, question, max_num=6, image_size=448):
    all_responses = []

    for image_path in imagePaths:
        # Load and preprocess the image
        pixel_values = load_image(image_path, max_num=max_num, input_size=image_size).to(torch.bfloat16).cuda()

        # Configuration for text generation
        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        # Ask the question
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        all_responses.append((question, response))

    return all_responses

# Function to initialize the model and tokenizer
def initialize_model_and_tokenizer(path="OpenGVLab/InternVL-Chat-V1-5"):
    model = (
        AutoModel.from_pretrained(
            path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        .eval()
        .cuda()
    )
    
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    
    return model, tokenizer

# Function for single-round single-image conversation
def single_round_single_image_conversation(model, tokenizer, pixel_values, question, generation_config):
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(question, response)
    return response

# Function for multi-round single-image conversation
def multi_round_single_image_conversation(model, tokenizer, pixel_values, questions, generation_config):
    history = None
    responses = []
    for question in questions:
        response, history = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=history,
            return_history=True,
        )
        print(question, response)
        responses.append(response)
    return responses

# Function for multi-round multi-image conversation
def multi_round_multi_image_conversation(model, tokenizer, pixel_values, questions, generation_config):
    history = None
    responses = []
    for question in questions:
        response, history = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=history,
            return_history=True,
        )
        print(question, response)
        responses.append(response)
    return responses

# Function for batch inference (single image per sample)
def batch_inference(model, tokenizer, pixel_values_list, questions, generation_config):
    image_counts = [pixel_values.size(0) for pixel_values in pixel_values_list]
    pixel_values = torch.cat(pixel_values_list, dim=0)
    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        image_counts=image_counts,
        questions=questions,
        generation_config=generation_config,
    )
    for question, response in zip(questions, responses):
        print(question)
        print(response)
    return responses

# Example usage
model, tokenizer = initialize_model_and_tokenizer()
imagePaths = ["./examples/image1.jpg", "./examples/image2.jpg"]
pixel_values_list = [load_image(image_path, max_num=6).to(torch.bfloat16).cuda() for image_path in imagePaths]
generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

# Single-round single-image conversation
single_round_single_image_conversation(model, tokenizer, pixel_values_list[0], "请详细描述图片", generation_config)

# Multi-round single-image conversation
multi_round_single_image_conversation(model, tokenizer, pixel_values_list[0], ["请详细描述图片", "请根据图片写一首诗"], generation_config)

# Multi-round multi-image conversation
multi_round_multi_image_conversation(model, tokenizer, torch.cat(pixel_values_list, dim=0), ["详细描述这两张图片", "这两张图片的相同点和区别分别是什么"], generation_config)

# Batch inference
batch_inference(model, tokenizer, pixel_values_list, ["Describe the image in detail."] * len(imagePaths), generation_config)