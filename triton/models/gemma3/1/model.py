import os
import numpy as np
from PIL import Image
import torch
import triton_python_backend_utils as pb_utils
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
import HyperParameters as hp

MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH")

REQUIRED_PROCESSOR_FILES = (
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)


def _missing_model_files(model_path: str):
    if not model_path or not os.path.isdir(model_path):
        return list(REQUIRED_PROCESSOR_FILES) + ["model*.safetensors"]
    missing = [
        name
        for name in REQUIRED_PROCESSOR_FILES
        if not os.path.isfile(os.path.join(model_path, name))
    ]
    has_weights = (
        os.path.isfile(os.path.join(model_path, "model.safetensors"))
        or os.path.isfile(os.path.join(model_path, "model.safetensors.index.json"))
        or any(
            name.startswith("model-") and name.endswith(".safetensors")
            for name in os.listdir(model_path)
        )
    )
    if not has_weights:
        missing.append("model*.safetensors")
    return missing


def _as_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_prompts(raw):
    return [_as_str(value) for value in np.asarray(raw).reshape(-1)]


def _images_from_tensor(image_np, batch_size):
    array = np.asarray(image_np)
    if array.ndim == 3:
        array = array[None, ...]
    if array.shape[0] == 1 and batch_size > 1:
        return [Image.fromarray(array[0], mode="RGB") for _ in range(batch_size)]
    return [
        Image.fromarray(array[i], mode="RGB") for i in range(batch_size)
    ]


def _answer_tensor(texts):
    """``answer`` is STRING dims [1], so batched shape is [B, 1]."""
    return pb_utils.Tensor(
        "answer",
        np.array([[text.encode("utf-8")] for text in texts], dtype=object),
    )


def _user_message(image, prompt_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


class TritonPythonModel:
    def initialize(self, args):
        if not MODEL_PATH:
            raise ValueError("GEMMA_MODEL_PATH is not set")

        missing = _missing_model_files(MODEL_PATH)
        if missing:
            raise FileNotFoundError(
                f"Incomplete Gemma model at {MODEL_PATH}; missing: {missing}. "
                "Delete the Triton pod to clear emptyDir and let entrypoint re-download "
                "(HF_TOKEN must have access to google/gemma-3-4b-it)."
            )

        # load GEMMA processor
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
            clean_up_tokenization_spaces=True,
        )
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"

        # load the GEMMA3 model
        if torch.cuda.is_available():
            gpu_card = 0
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                torch_dtype="auto",
                device_map={"": gpu_card} # assigns layers to GPU
            ).eval()
        else:
            #CPU
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                torch_dtype="auto",
            ).eval()

        self.device = next(self.model.parameters()).device

    def _generate_conversations(self, conversations):
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=hp.max_new_tokens,
                early_stopping=hp.early_stopping,
                do_sample=hp.do_sample,
                num_beams=hp.num_beams,
            )
        new_tokens = generated[:, prompt_len:]
        return [
            self.processor.decode(row, skip_special_tokens=True)
            for row in new_tokens
        ]

    def _generate_one(self, conversation):
        try:
            return self._generate_conversations([conversation])[0]
        except Exception as e:
            print("Error during model.generate or decode:", e)
            return f"ERROR in generate/decode: {e}"

    def execute(self, requests):
        """
        Collate samples across ``requests`` and run one padded generate when
        possible. A failed batch falls back to per-sample generate so one bad
        image does not fail the whole dynamic batch.
        """
        request_sizes = []
        conversations = []
        parse_errors = []

        for request in requests:
            try:
                prompts = _decode_prompts(
                    pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()
                )
                images = _images_from_tensor(
                    pb_utils.get_input_tensor_by_name(request, "image").as_numpy(),
                    len(prompts),
                )
            except Exception as e:
                print("Unexpected error parsing Gemma request:", e)
                request_sizes.append(1)
                conversations.append(None)
                parse_errors.append(f"UNEXPECTED ERROR: {e}")
                continue

            request_sizes.append(len(prompts))
            for image, prompt_text in zip(images, prompts):
                try:
                    conversations.append(_user_message(image, prompt_text))
                    parse_errors.append(None)
                except Exception as e:
                    print("Error building Gemma chat message:", e)
                    conversations.append(None)
                    parse_errors.append(f"UNEXPECTED ERROR: {e}")

        valid_idx = [i for i, conv in enumerate(conversations) if conv is not None]
        answers = list(parse_errors)

        if valid_idx:
            valid_conversations = [conversations[i] for i in valid_idx]
            try:
                generated = self._generate_conversations(valid_conversations)
                for i, text in zip(valid_idx, generated):
                    answers[i] = text
            except Exception as e:
                print("Error in batched apply_chat_template/generate:", e)
                for i in valid_idx:
                    answers[i] = self._generate_one(conversations[i])

        responses = []
        offset = 0
        for batch_size in request_sizes:
            end = offset + batch_size
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[_answer_tensor(answers[offset:end])]
                )
            )
            offset = end
        return responses

    def finalize(self):
        pass
