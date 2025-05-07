import torch
import gc
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

# Configuration settings
DEFAULT_OCR_INSTRUCTION = "Read all text in the image. Do not add any comments."
DEFAULT_SUMMARY_INSTRUCTION = "Summarize text in Polish. Make it no more than 4 sentences."

# Model paths
OCR_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
SUMMARY_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

def get_image_metadata(image_path):
    """Retrieve image width and height."""
    with Image.open(image_path) as img:
        return img.width, img.height

def get_bnb4_config():
    """Return BitsAndBytes configuration for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Fixed typo: loat16 -> float16
        bnb_4bit_use_double_quant=True,
    )

def load_model_vl(model_name):
    """Load vision-language model with optimized settings."""
    return AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="balanced",
        trust_remote_code=True,
    )

def load_model_tx(model_name):
    """Load text model with optimized settings."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

def load_processor(model_name):
    """Load processor for the specified model."""
    return AutoProcessor.from_pretrained(
        model_name, 
        device_map="auto"
    )

def memory_stats():
    """Print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_cached() / 1024**2
    print(f"GPU Memory Allocated: {allocated:.2f} MB")
    print(f"GPU Memory Cached: {cached:.2f} MB")

def process_image_to_text(image_path, model, processor, command):
    """Extract text from an image using the vision-language model."""
    # Prepare chat messages
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are OCR system for text recognition."}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": command},
            ],
        },
    ]

    # Convert messages into model input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)  # We don't use video inputs here

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate output text
    generated_ids = model.generate(**inputs, max_new_tokens=32000)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode text and process newlines
    output_texts = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    output_text = "\n".join(output_texts).replace("\\n", "\n")

    return output_text

def process_text_to_text(text, model, model_name, instruction):
    """Process text using the text model with provided instruction."""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text}
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Convert messages into model input
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Generate output text
    generated_ids = model.generate(**model_inputs, max_new_tokens=100000)
    generated_ids_trimmed = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    output_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output_text

def clean_resources(model=None, processor=None):
    """Clean up resources to free memory."""
    if model is not None:
        del model
    if processor is not None:
        del processor
    torch.cuda.empty_cache()
    gc.collect()

def ocr_image(image_file, instruction=DEFAULT_OCR_INSTRUCTION):
    """Main OCR function to extract text from an image."""
    print(f"Processing file: {image_file}")
    print(f"Instruction: {instruction}")
    
    try:
        model = load_model_vl(OCR_MODEL_PATH)
        processor = load_processor(OCR_MODEL_PATH)
        
        result_text = process_image_to_text(image_file, model, processor, instruction)
        
        # Clean up resources
        clean_resources(model, processor)
        return result_text
        
    except Exception as e:
        print(f"Error in OCR process: {str(e)}")
        clean_resources()
        return f"OCR Error: {str(e)}"

def summarize_text(text, instruction=DEFAULT_SUMMARY_INSTRUCTION):
    """Summarize text using a text model with provided instruction."""
    print(f"Summarizing text with instruction: {instruction}")
    
    try:
        model = load_model_tx(SUMMARY_MODEL_PATH)
        
        result_text = process_text_to_text(text, model, SUMMARY_MODEL_PATH, instruction)
        
        # Clean up resources
        clean_resources(model)
        return result_text
        
    except Exception as e:
        print(f"Error in summarization process: {str(e)}")
        clean_resources()
        return f"Summarization Error: {str(e)}"
