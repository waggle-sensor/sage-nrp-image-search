import os
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from transformers import CLIPProcessor, CLIPModel

MODEL_PATH = os.environ.get("CLIP_MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        """
        Load CLIP’s processor and model in one shot.
        """
        # Load CLIPProcessor and CLIPModel from Hugging Face
        self.processor = CLIPProcessor.from_pretrained(MODEL_PATH)
        self.model = CLIPModel.from_pretrained(MODEL_PATH,use_safetensors=True).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()

        # Dynamically pick up the projection dimension (e.g. 1024 for ViT-H/14)
        self.embedding_dim = self.model.config.projection_dim
        self.device = next(self.model.parameters()).device

    def execute(self, requests):
        """
        For each incoming request:
          - If there’s a non‐empty “text” tensor, run get_text_features(...) → shape [B, D]
          - If there’s a non‐zero “image” tensor, run get_image_features(...) → shape [B, D]
        """
        responses = []

        for request in requests:
            # Default outputs (batch-size is 1 for simplicity)
            text_embeddings = np.zeros((1, self.embedding_dim), dtype=np.float32)
            image_embeddings = np.zeros((1, self.embedding_dim), dtype=np.float32)

            # ── 1) TEXT BRANCH ───────────────────────────────────────
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            raw_texts = text_tensor.as_numpy()  # shape: [1] of bytes
            batch_texts = [t.decode("utf-8").strip() for t in raw_texts]
            if not all(text == "" for text in batch_texts):
                encoded = self.processor(
                    text=batch_texts,
                    padding=True,         # pad all samples to the longest in the batch
                    truncation=True,      # truncate anything longer than the model’s max length
                    return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    feats = self.model.get_text_features(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )
                text_embeddings = feats.cpu().numpy().astype(np.float32)

            # ── 2) IMAGE BRANCH ───────────────────────────────────────
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            image_np = image_tensor.as_numpy()
            if not np.all(image_np == 0):
                inputs = self.processor( images=image_np, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                image_embeddings = feats.cpu().numpy().astype(np.float32)

            # ── 3) PACKAGE THE OUTPUTS ────────────────────────────────
            output_tensors = []
            text_out = pb_utils.Tensor("text_embedding", text_embeddings)
            img_out = pb_utils.Tensor("image_embedding", image_embeddings)
            output_tensors.extend([text_out, img_out])
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def finalize(self):
        """No special cleanup required."""
        pass
