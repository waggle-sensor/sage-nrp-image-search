import os
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from transformers import AlignProcessor, AlignModel

MODEL_PATH = os.environ.get("ALIGN_MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        """
        Load ALIGN’s processor and model in one shot.
        """
        self.embedding_dim = 256  # ALIGN embedding size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # AlignProcessor handles BOTH text-tokenization and image-preprocessing
        self.processor = AlignProcessor.from_pretrained(MODEL_PATH)

        # ALIGN model itself (supports get_text_features & get_image_features)
        self.model = AlignModel.from_pretrained(MODEL_PATH).to(self.device)
        self.model.eval()

    def execute(self, requests):
        """
        For each incoming request:
          - If it has a “text” tensor, run get_text_features(...) → shape [B, D]
          - If it has an “image” tensor, run get_image_features(...) → shape [B, D]
        """
        responses = []

        for request in requests:
            # Default outputs
            text_embeddings = np.zeros((1, self.embedding_dim), dtype=np.float32)
            image_embeddings = np.zeros((1, self.embedding_dim), dtype=np.float32)

            # ── 1) TEXT BRANCH ───────────────────────────────────────
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            raw_texts = text_tensor.as_numpy()
            batch_texts = [t.decode("utf-8").strip() for t in raw_texts]
            if not all(text == "" for text in batch_texts):
                # AlignProcessor will tokenize
                encoded = self.processor(text=batch_texts, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feats = self.model.get_text_features(**encoded)
                text_embeddings = feats.cpu().numpy().astype(np.float32)

            # ── 2) IMAGE BRANCH ───────────────────────────────────────
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            image_np = image_tensor.as_numpy() 
            if not np.all(image_np == 0):               
                inputs = self.processor(images=image_np, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)  # shape: [B, D]
                image_embeddings = feats.cpu().numpy().astype(np.float32)  # [B, D]
            
            # ── 3) PACKAGE THE OUTPUTS ────────────────────────────────
            output_tensors = []
            # text_embeddings: shape [B, D]
            text_tensor_out = pb_utils.Tensor("text_embedding", text_embeddings)
            output_tensors.append(text_tensor_out)
            # image_embeddings: shape [B, D]
            img_tensor_out = pb_utils.Tensor("image_embedding", image_embeddings)
            output_tensors.append(img_tensor_out)
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def finalize(self):
        """No special cleanup required."""
        pass
