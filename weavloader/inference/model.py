'''This file contains the code to talk to Triton Inference Server'''

import logging
from collections import OrderedDict
import tritonclient.grpc as TritonClient
import numpy as np
from . import model_config as hp
import json

def florence2_run_model(triton_client, task_prompt, image, text_input=""):
    """
    takes in a task prompt and image, returns an answer using florence2 base model
    """
    # Prepare inputs for Triton
    image_width, image_height = image.size
    image_np = np.array(image).astype(np.float32)
    task_prompt_bytes = task_prompt.encode("utf-8")
    text_input_bytes = text_input.encode("utf-8")

    # Prepare inputs & outputs for Triton
    # NOTE: if you enable max_batch_size, leading number is batch size, example [1,1] 1 is batch size
    inputs = [
        TritonClient.InferInput("image", [image_height, image_width, 3], "FP32"),
        TritonClient.InferInput("prompt", [1], "BYTES"),
        TritonClient.InferInput("text_input", [1], "BYTES"),
        TritonClient.InferInput("image_width", [1], "INT32"),
        TritonClient.InferInput("image_height", [1], "INT32")
    ]
    outputs = [
        TritonClient.InferRequestedOutput("answer")
    ]

    # Add tensors
    inputs[0].set_data_from_numpy(image_np)
    inputs[1].set_data_from_numpy(np.array([task_prompt_bytes], dtype="object"))
    inputs[2].set_data_from_numpy(np.array([text_input_bytes], dtype="object"))
    inputs[3].set_data_from_numpy(np.array([image_width], dtype="int32"))
    inputs[4].set_data_from_numpy(np.array([image_height], dtype="int32"))

    # Perform inference
    try:
        response = triton_client.infer(model_name="florence2base", inputs=inputs, outputs=outputs)

        # Get the result
        answer = response.as_numpy("answer")[0]
        answer_str = answer.decode("utf-8")

        # Convert the JSON string to a dictionary
        answer_dict = json.loads(answer_str)

        return answer_dict
    except Exception as e:
        logging.error(f"[MODEL] Error during Florence2 inference: {str(e)}")
        return None

def florence2_gen_caption(triton_client, image):
    """
    Generate image caption using florence2 base model
    """
    task_prompt = '<MORE_DETAILED_CAPTION>'

    description_text = florence2_run_model(triton_client, task_prompt, image)
    description_text = description_text[task_prompt]

    #takes those details from the setences and finds labels and boxes in the image
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    boxed_descriptions = florence2_run_model(triton_client, task_prompt, image, description_text)

    #only prints out labels not bboxes
    descriptions = boxed_descriptions[task_prompt]['labels']
    logging.info(f'[MODEL] Labels Generated: {descriptions}')

    #finds other things in the image that the description did not explicitly say
    task_prompt = '<DENSE_REGION_CAPTION>'
    labels = florence2_run_model(triton_client, task_prompt, image)

    #only prints out labels not bboxes
    printed_labels = labels[task_prompt]['labels']

    # Join description_text into a single string
    description_text_joined = "".join(description_text)

    #makes unique list of labels and adds commas
    label_list = descriptions + printed_labels
    unique_labels = list(OrderedDict.fromkeys(label_list))
    labels = ", ".join(unique_labels)

    # Combine all lists into one list
    combined_list = ["DESCRIPTION:"] + [description_text_joined] + ["LABELS:"] + [labels]

    # Join the unique items into a single string with spaces between them
    final_description = " ".join(combined_list)

    logging.info(f'[MODEL] Final Generated Description: {final_description}')
    return final_description

def get_colbert_embedding(triton_client, text):
    """
    Embed text using ColBERT encoder served via Triton Inference Server.
    Returns token-level embeddings of shape [num_tokens, 128]
    """
    # Prepare input
    text_bytes = text.encode("utf-8")
    input_tensor = np.array([text_bytes], dtype="object")  # batch size = 1

    # Prepare inputs & outputs for Triton
    # NOTE: if you enable max_batch_size, leading number is batch size, example [1,1] 1 is batch size
    inputs = [
        TritonClient.InferInput("text", input_tensor.shape, "BYTES")
    ]
    outputs = [
        TritonClient.InferRequestedOutput("embedding"),
        TritonClient.InferRequestedOutput("token_lengths")
    ]

    # Add tensors
    inputs[0].set_data_from_numpy(input_tensor)

    # Run inference
    try:
        results = triton_client.infer(model_name="colbert", inputs=inputs, outputs=outputs)

        # Retrieve and reshape output
        emb_flat = results.as_numpy("embedding")           # shape: (1, max_len * 128)
        token_lengths = results.as_numpy("token_lengths")  # shape: (1,)
        num_tokens = token_lengths[0]

        # Reshape and unpad
        emb_3d = emb_flat.reshape(1, -1, 128)
        token_embeddings = emb_3d[0, :num_tokens, :]  # shape: [num_tokens, 128]
    except Exception as e:
        logging.error(f"[MODEL] Error during Colbert inference: {str(e)}")
        return None

    return token_embeddings

def fuse_embeddings( img_emb: np.ndarray, txt_emb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Given two L2-normalized vectors img_emb and txt_emb (shape (D,)), 
    returns their weighted sum (alpha * img + (1-alpha) * txt), re-normalized to unit norm.
    """
    if img_emb.shape != txt_emb.shape:
        raise ValueError("img_emb and txt_emb must have the same dimension")

    # Weighted sum
    combined = alpha * img_emb + (1.0 - alpha) * txt_emb

    # Re-normalize
    norm = np.linalg.norm(combined)
    if norm == 0.0:
        # Edge case: if they cancel out exactly (unlikely), fall back to text alone
        return txt_emb.copy()
    return (combined / norm).astype(np.float32)

def get_allign_embeddings(triton_client, text, image=None):
    """
    Embed text and image using ALIGN encoder served via Triton Inference Server.
    Returns one fused embedding created from both modalities.
    """
    # --- 1. Prepare Inputs ---
    text_bytes = text.encode("utf-8")
    text_np = np.array([text_bytes], dtype="object")

    # Fallback image shape (e.g., placeholder 1x1 RGB)
    if image is not None:
        image_np = np.array(image).astype(np.float32)
    else:
        image_np = np.zeros((1, 1, 3), dtype=np.float32)

    # Create Triton input objects
    inputs = [
        TritonClient.InferInput("text", [1], "BYTES"),
        TritonClient.InferInput("image", list(image_np.shape), "FP32")
    ]

    inputs[0].set_data_from_numpy(text_np)
    inputs[1].set_data_from_numpy(image_np)

    outputs = [
        TritonClient.InferRequestedOutput("text_embedding"),
        TritonClient.InferRequestedOutput("image_embedding")
    ]

    # --- 2. Inference Call ---
    try:
        results = triton_client.infer(model_name="align", inputs=inputs, outputs=outputs)
        text_embedding = results.as_numpy("text_embedding")[0]
        image_embedding = results.as_numpy("image_embedding")[0]
    except Exception as e:
        logging.error(f"[MODEL] Error during ALIGN inference: {str(e)}")
        return None

    # --- 3. Fuse Embeddings ---
    if image is not None:
        embedding = fuse_embeddings(image_embedding, text_embedding, alpha=hp.align_alpha)
    else:
        embedding = text_embedding

    return embedding

def get_clip_embeddings(triton_client, text, image=None):
    """
    Embed text and image using CLIP encoder served via Triton Inference Server.
    Returns one fused embedding created from both modalities.
    """
    # --- 1. Prepare Inputs ---
    text_bytes = text.encode("utf-8")
    text_np = np.array([text_bytes], dtype="object")

    # Fallback image shape (e.g., placeholder 1x1 RGB)
    if image is not None:
        image_np = np.array(image).astype(np.float32)
    else:
        image_np = np.zeros((1, 1, 3), dtype=np.float32)

    # Create Triton input objects
    inputs = [
        TritonClient.InferInput("text", [1], "BYTES"),
        TritonClient.InferInput("image", list(image_np.shape), "FP32")
    ]

    inputs[0].set_data_from_numpy(text_np)
    inputs[1].set_data_from_numpy(image_np)

    outputs = [
        TritonClient.InferRequestedOutput("text_embedding"),
        TritonClient.InferRequestedOutput("image_embedding")
    ]

    # --- 2. Inference Call ---
    try:
        results = triton_client.infer(model_name="clip", inputs=inputs, outputs=outputs)
        text_embedding = results.as_numpy("text_embedding")[0]
        image_embedding = results.as_numpy("image_embedding")[0]
    except Exception as e:
        logging.error(f"[MODEL] Error during CLIP inference: {str(e)}")
        return None

    # --- 3. Fuse Embeddings ---
    if image is not None:
        embedding = fuse_embeddings(image_embedding, text_embedding, alpha=hp.clip_alpha)
    else:
        embedding = text_embedding

    return embedding

def qwen2_5_run_model(triton_client, image, task_prompt=hp.qwen2_5_prompt):
    """
    takes in a task prompt and image, returns an answer using Qwen2.5-VL model
    """
    # Prepare inputs for Triton
    image_width, image_height = image.size
    image_np = np.array(image).astype(np.uint8)
    task_prompt_bytes = task_prompt.encode("utf-8")

    # Prepare inputs & outputs for Triton
    # NOTE: if you enable max_batch_size, leading number is batch size, example [1,1] 1 is batch size
    inputs = [
        TritonClient.InferInput("image", [image_height, image_width, 3], "UINT8"),
        TritonClient.InferInput("prompt", [1], "BYTES"),
    ]
    outputs = [
        TritonClient.InferRequestedOutput("answer")
    ]

    # Add tensors
    inputs[0].set_data_from_numpy(image_np)
    inputs[1].set_data_from_numpy(np.array([task_prompt_bytes], dtype="object"))

    # Perform inference
    try:
        response = triton_client.infer(model_name="qwen2_5_vl", inputs=inputs, outputs=outputs)

        # Get the result
        answer = response.as_numpy("answer")[0]
        answer_str = answer.decode("utf-8")

        logging.info(f'[MODEL] Final Generated Description: {answer_str}')
        return answer_str
    except Exception as e:
        logging.error(f"[MODEL] Error during Qwen2.5-VL inference: {str(e)}")
        return None
    
def gemma3_run_model(triton_client, image, task_prompt=hp.gemma3_prompt):
    """
    takes in a task prompt and image, returns an answer using gemma3 model
    """
    # Prepare inputs for Triton
    image_width, image_height = image.size
    image_np = np.array(image).astype(np.uint8)
    task_prompt_bytes = task_prompt.encode("utf-8")

    # Prepare inputs & outputs for Triton
    # NOTE: if you enable max_batch_size, leading number is batch size, example [1,1] 1 is batch size
    inputs = [
        TritonClient.InferInput("image", [image_height, image_width, 3], "UINT8"),
        TritonClient.InferInput("prompt", [1], "BYTES"),
    ]
    outputs = [
        TritonClient.InferRequestedOutput("answer")
    ]

    # Add tensors
    inputs[0].set_data_from_numpy(image_np)
    inputs[1].set_data_from_numpy(np.array([task_prompt_bytes], dtype="object"))

    # Perform inference
    try:
        response = triton_client.infer(model_name="gemma3", inputs=inputs, outputs=outputs)

        # Get the result
        answer = response.as_numpy("answer")[0]
        answer_str = answer.decode("utf-8")

        logging.info(f'[MODEL] Final Generated Description: {answer_str}')
        return answer_str
    except Exception as e:
        logging.error(f"[MODEL] Error during Gemma3 inference: {str(e)}")
        return None
