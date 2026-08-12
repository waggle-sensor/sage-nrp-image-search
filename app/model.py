'''This file contains the code to talk to Triton Inference Server'''

import logging
import tritonclient.grpc as TritonClient
import numpy as np
import HyperParameters as hp


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

def _infer_clip(triton_client, text, image=None, request_logit_scale: bool = False):
    """
    Run Triton CLIP and return (text_embedding, image_embedding[, logit_scale]).
    On failure returns (None, None) or (None, None, None) if request_logit_scale.
    """
    text_bytes = text.encode("utf-8")
    text_np = np.array([text_bytes], dtype="object")

    if image is not None:
        image_np = np.array(image).astype(np.float32)
    else:
        image_np = np.zeros((1, 1, 3), dtype=np.float32)

    inputs = [
        TritonClient.InferInput("text", [1], "BYTES"),
        TritonClient.InferInput("image", list(image_np.shape), "FP32"),
    ]
    inputs[0].set_data_from_numpy(text_np)
    inputs[1].set_data_from_numpy(image_np)

    outputs = [
        TritonClient.InferRequestedOutput("text_embedding"),
        TritonClient.InferRequestedOutput("image_embedding"),
    ]
    if request_logit_scale:
        outputs.append(TritonClient.InferRequestedOutput("logit_scale"))

    try:
        results = triton_client.infer(model_name="clip", inputs=inputs, outputs=outputs)
        text_embedding = results.as_numpy("text_embedding")[0]
        image_embedding = results.as_numpy("image_embedding")[0]
        if request_logit_scale:
            logit_scale = float(results.as_numpy("logit_scale").reshape(-1)[0])
            return text_embedding, image_embedding, logit_scale
        return text_embedding, image_embedding
    except Exception as e:
        logging.error(f"Error during CLIP inference: {str(e)}")
        if request_logit_scale:
            return None, None, None
        return None, None


def get_clip_embeddings(triton_client, text, image=None):
    """
    Embed text and image using CLIP encoder served via Triton Inference Server.
    Returns one fused embedding created from both modalities.

    Production ingest/query no longer fuse at index time; see
    ``get_clip_embedding_pair`` in weavloader. This helper is kept for
    experiments that still want a single combined vector.
    """
    text_embedding, image_embedding = _infer_clip(triton_client, text, image)
    if text_embedding is None:
        return None

    if image is not None:
        return fuse_embeddings(image_embedding, text_embedding, alpha=hp.clip_alpha)
    return text_embedding


def clip_image_text_score(triton_client, query: str, image) -> float:
    """
    CLIP similarity between a text query and an image via Triton.

    Matches Hugging Face CLIPModel logits_per_image for a single pair:
      L2-normalize image/text embeddings, then multiply cosine by exp(logit_scale).
    """
    text_embedding, image_embedding, logit_scale = _infer_clip(
        triton_client, query, image, request_logit_scale=True
    )
    if text_embedding is None or image_embedding is None or logit_scale is None:
        return 0.0

    text_emb = np.asarray(text_embedding, dtype=np.float32)
    image_emb = np.asarray(image_embedding, dtype=np.float32)

    text_norm = np.linalg.norm(text_emb)
    image_norm = np.linalg.norm(image_emb)
    if text_norm == 0.0 or image_norm == 0.0:
        return 0.0

    text_emb = text_emb / text_norm
    image_emb = image_emb / image_norm
    cosine = float(np.dot(image_emb, text_emb))
    return cosine * float(logit_scale)
