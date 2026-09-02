import logging
import os
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from transformers import CLIPProcessor, CLIPModel


def _as_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def _decode_texts(raw):
    """Flatten a [B] or [B, 1] STRING tensor to a list of Python str."""
    return [_as_str(value) for value in np.asarray(raw).reshape(-1)]


def _split_images(image_np, batch_size):
    """Yield ``batch_size`` HWC image arrays from a 3-D or 4-D tensor."""
    array = np.asarray(image_np)
    if array.ndim == 3:
        array = array[None, ...]
    if array.shape[0] == 1 and batch_size > 1:
        return [array[0] for _ in range(batch_size)]
    return [array[i] for i in range(batch_size)]


class TritonPythonModel:
    def initialize(self, args):
        """
        Load CLIP's processor and model in one shot.
        """
        model_path = os.environ.get("CLIP_MODEL_PATH", "/models/clip")
        if not model_path or not os.path.isdir(model_path) or not os.listdir(model_path):
            raise pb_utils.TritonModelException(
                f"CLIP_MODEL_PATH '{model_path}' is missing or empty. "
                "Ensure the entrypoint downloaded the model or mount weights at this path."
            )

        use_safetensors = os.environ.get("USE_SAFETENSORS", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        logging.info(
            "Loading CLIP model from %s (use_safetensors=%s)", model_path, use_safetensors
        )
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(
            model_path, use_safetensors=use_safetensors
        ).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()

        # Dynamically pick up the projection dimension (e.g. 1024 for ViT-H/14)
        self.embedding_dim = self.model.config.projection_dim
        self.device = next(self.model.parameters()).device

    def execute(self, requests):
        """
        Collate every sample across ``requests`` into one CLIP text forward
        and one CLIP image forward, then scatter embeddings per request.

        Dynamic batching may pass several requests (ragged images) or one
        request whose leading dim is already the batch size.
        """
        texts = []
        images = []
        request_sizes = []

        for request in requests:
            batch_texts = _decode_texts(
                pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            )
            batch_images = _split_images(
                pb_utils.get_input_tensor_by_name(request, "image").as_numpy(),
                len(batch_texts),
            )
            request_sizes.append(len(batch_texts))
            texts.extend(batch_texts)
            images.extend(batch_images)

        n = len(texts)
        text_embeddings = np.zeros((n, self.embedding_dim), dtype=np.float32)
        image_embeddings = np.zeros((n, self.embedding_dim), dtype=np.float32)

        text_idx = [i for i, text in enumerate(texts) if text]
        image_idx = [i for i, image in enumerate(images) if not np.all(image == 0)]

        if text_idx:
            encoded = self.processor(
                text=[texts[i] for i in text_idx],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                feats = self.model.get_text_features(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
            text_embeddings[text_idx] = feats.cpu().numpy().astype(np.float32)

        if image_idx:
            encoded = self.processor(
                images=[images[i] for i in image_idx],
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(
                    pixel_values=encoded["pixel_values"]
                )
            image_embeddings[image_idx] = feats.cpu().numpy().astype(np.float32)

        logit_scale = (
            self.model.logit_scale.detach()
            .exp()
            .float()
            .cpu()
            .numpy()
            .reshape(1, 1)
            .astype(np.float32)
        )

        responses = []
        offset = 0
        for batch_size in request_sizes:
            end = offset + batch_size
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "text_embedding", text_embeddings[offset:end]
                        ),
                        pb_utils.Tensor(
                            "image_embedding", image_embeddings[offset:end]
                        ),
                        pb_utils.Tensor(
                            "logit_scale", np.repeat(logit_scale, batch_size, axis=0)
                        ),
                    ]
                )
            )
            offset = end

        return responses

    def finalize(self):
        """No special cleanup required."""
        pass
