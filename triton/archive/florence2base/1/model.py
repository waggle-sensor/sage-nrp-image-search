import numpy as np
import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import triton_python_backend_utils as pb_utils
import json
import HyperParameters as hp

MODEL_PATH = os.environ.get("MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        # Load the Florence 2 processor
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
            clean_up_tokenization_spaces=True
        )

        # Load the Florence 2 model for inference
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Check if GPU is available and move the model to GPU if possible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move the model to GPU if available

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get inputs from request
            image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()           
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()
            txtinput_tensor = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()
            image_width = pb_utils.get_input_tensor_by_name(request, "image_width").as_numpy()[0]
            image_height = pb_utils.get_input_tensor_by_name(request, "image_height").as_numpy()[0]

            # Decode the strings
            task_prompt = prompt_tensor[0].decode("utf-8")
            txtinput = txtinput_tensor[0].decode("utf-8") if txtinput_tensor.size > 0 else None

            # Add txt input if provided
            if txtinput is None:
                prompt = task_prompt
            else:
                prompt = task_prompt + txtinput

            # Preprocess the image and text using Florence 2 processor
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

            # Run inference using the Florence 2 model
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=hp.max_new_tokens,
                early_stopping=hp.early_stopping,
                do_sample=hp.do_sample,
                num_beams=hp.num_beams,
            )

            # Decode the generated ids into text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # Post-process the generated text
            answer_dict = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image_width, image_height)
            )

            # Convert the dictionary to a string
            answer_str = json.dumps(answer_dict)

            # Encode the answer string into bytes
            answer = answer_str.encode("utf-8")

            # Prepare the final parsed answer as a response
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("answer", np.array([answer], dtype=object))
            ])
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass  # Cleanup if necessary
