import numpy as np
import os
import triton_python_backend_utils as pb_utils
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = os.environ.get("COLBERT_MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the Triton Python model by loading a Hugging Face transformer model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model from local path or HF hub
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(self.device)
        self.model.eval()

    def execute(self, requests):
        """
        Perform inference using the transformer encoder and return full token-level embeddings.
        """
        responses = []

        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            if text_tensor is None:
                raise ValueError("Input 'text' tensor is required.")
            text = text_tensor.as_numpy()[0].decode("utf-8")

            # Tokenize and move to device
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # [tokens, dim]

            # Prepare output (pad to fixed shape across batch, Triton doesn't support jagged arrays)
            embeddings = [token_embeddings]  # list for batch compatibility
            max_len = max(e.shape[0] for e in embeddings)
            batch_size = len(embeddings)
            emb_dim = embeddings[0].shape[1]
            padded = np.zeros((batch_size, max_len, emb_dim), dtype=np.float32)
            for i, emb in enumerate(embeddings):
                padded[i, :emb.shape[0], :] = emb

            token_lengths = np.array([e.shape[0] for e in embeddings], dtype=np.int32)
            output_tensor = pb_utils.Tensor("embedding", padded)
            length_tensor = pb_utils.Tensor("token_lengths", token_lengths)

            responses.append(pb_utils.InferenceResponse(
                output_tensors=[output_tensor, length_tensor]
            ))

        return responses

    def finalize(self):
        """Cleanup resources."""
        pass