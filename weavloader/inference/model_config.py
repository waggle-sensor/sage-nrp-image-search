import os
'''This file contains the hyper parameters that can be changed to fine tune
the system. '''

nrp_enable_thinking = (
    os.environ.get("NRP_ENABLE_THINKING", "false").lower() in ("1", "true", "yes")
)
# Caption LLM image caps: LLM_MAX_IMAGE_* / LLM_IMAGE_BYTE_LIMITING (see inference/image_utils.py)
align_alpha = 0.7
clip_alpha = 0.7  # Used by fuse_embeddings() only; ingest stores caption/image vectors separately.
default_prompt="""
role:
You are a world-class Scientific Image Captioning Expert.

context:
You will be shown a scientific image captured by edge devices. Your goal is to analyze its content and significance in detail. 

task:
Generate exactly one scientifically detailed caption that accurately describes what is visible in the image and its scientific relevance. 
Make it as detailed as possible. Also extract text and numbers from the images.

constraints:
- Only return:
  1. A single caption.
  2. a list of 15 keywords relevant to the image.
- Do not include any additional text, explanations, or formatting.

format:
  caption: <your_scientific_caption_here>
  keywords: <keyword1>, <keyword2>, ...
"""
caption_model_prompt = os.environ.get("CAPTION_MODEL_PROMPT", default_prompt)