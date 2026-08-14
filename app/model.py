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


def get_clip_query_embedding(triton_client, text):
    """
    Encode query text once via Triton CLIP.

    Returns (text_embedding, logit_scale) where logit_scale is exp(model.logit_scale),
    matching HF CLIPModel logits_per_image. On failure returns (None, None).
    """
    text_embedding, _, logit_scale = _infer_clip(
        triton_client, text, image=None, request_logit_scale=True
    )
    if text_embedding is None or logit_scale is None:
        return None, None
    return text_embedding, logit_scale


def clip_logits_per_image(text_embedding, image_vectors, logit_scale) -> np.ndarray:
    """
    Vectorized CLIP logits_per_image: cosine(image, text) * logit_scale.

    ``image_vectors`` is (N, D); ``text_embedding`` is (D,).
    Rows with missing or zero vectors score 0.0.
    """
    text = np.asarray(text_embedding, dtype=np.float32).reshape(-1)
    images = np.asarray(image_vectors, dtype=np.float32)
    if images.ndim == 1:
        images = images.reshape(1, -1)
    n = images.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    text_norm = np.linalg.norm(text)
    if text_norm == 0.0:
        return scores
    text = text / text_norm
    img_norms = np.linalg.norm(images, axis=1)
    valid = np.isfinite(img_norms) & (img_norms > 0)
    if not np.any(valid):
        return scores
    images_n = images[valid] / img_norms[valid, None]
    scores[valid] = (images_n @ text) * float(logit_scale)
    return scores


def clip_image_text_score(
    triton_client,
    query: str,
    image,
    text_embedding=None,
    logit_scale=None,
) -> float:
    """
    CLIP similarity between a text query and an image via Triton.

    Matches Hugging Face CLIPModel logits_per_image for a single pair:
      L2-normalize image/text embeddings, then multiply cosine by exp(logit_scale).

    Pass precomputed ``text_embedding`` and ``logit_scale`` to skip the text tower
    (empty text is sent so Triton only runs get_image_features).
    """
    if text_embedding is not None and logit_scale is not None:
        _, image_embedding = _infer_clip(triton_client, "", image)
        if image_embedding is None:
            return 0.0
    else:
        text_embedding, image_embedding, logit_scale = _infer_clip(
            triton_client, query, image, request_logit_scale=True
        )
        if text_embedding is None or image_embedding is None or logit_scale is None:
            return 0.0

    scores = clip_logits_per_image(text_embedding, [image_embedding], logit_scale)
    return float(scores[0])
