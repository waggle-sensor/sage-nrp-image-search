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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        # load GEMMA processor
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
            clean_up_tokenization_spaces=True,
        )

        gpu_card = 0
        # load the GEMMA3 model
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype="auto",
            device_map={"": gpu_card} # assigns layers to GPU
        ).eval()

        # choose device
        self.device = torch.device(f"cuda:{gpu_card}" if torch.cuda.is_available() else "cpu")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # pull in image + prompt
                image_arr = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
                image = Image.fromarray(image_arr, mode='RGB')

                prompt_bytes = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
                prompt_text = prompt_bytes.decode("utf-8")

                # build message list for chat template
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]

                # apply GEMMA’s chat template
                try:
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(self.model.device, dtype=torch.bfloat16)
                except Exception as e:
                    print("Error in apply_chat_template:", e)
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("answer", np.array([f"ERROR in apply_chat_template: {e}"], dtype=object))]
                    ))
                    continue

                input_len = inputs["input_ids"].shape[-1]

                # generate
                try:
                    with torch.inference_mode():
                        generated = self.model.generate(
                            **inputs,
                            max_new_tokens=hp.max_new_tokens,
                            early_stopping=hp.early_stopping,
                            do_sample=hp.do_sample,
                            num_beams=hp.num_beams,
                        )
                        generation = generated[0][input_len:]
                        output_text = self.processor.decode(generation, skip_special_tokens=True)
                except Exception as e:
                    print("Error during model.generate or decode:", e)
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("answer", np.array([f"ERROR in generate/decode: {e}"], dtype=object))]
                    ))
                    continue

                # build Triton tensor
                answer_bytes = output_text.encode("utf-8")
                out_tensor = pb_utils.Tensor("answer", np.array([answer_bytes], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            except Exception as e:
                print("Unexpected error in execute:", e)
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("answer", np.array([f"UNEXPECTED ERROR: {e}"], dtype=object))]
                ))

        return responses

    def finalize(self):
        pass
