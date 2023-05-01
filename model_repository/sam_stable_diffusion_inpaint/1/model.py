import triton_python_backend_utils as pb_utils
from pathlib import Path
import torch.utils.dlpack as dlpack
import inspect
from transformers import CLIPTokenizer
from diffusers.schedulers import PNDMScheduler
import torch
import numpy as np
from typing import Dict, List, Union

from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple
import PIL

class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def prepare_image(image, transform, normalize=False):

    image = transform.apply_image(image)
    image = torch.as_tensor(image) 
    
    if len(image.shape) == 3:
        image = image.permute(2, 0, 1)

    h, w = image.shape[-2:]
    padh = transform.target_length - h
    padw = transform.target_length - w

    if normalize == True:
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        image = (image - pixel_mean) / pixel_std
        image = F.pad(image, (0, padw, 0, padh)).unsqueeze(0)
    else:
        image = F.pad(image, (0, padw, 0, padh))
        image = image.cpu().numpy()
        
    return image


def prepare_mask_and_masked_image(image, mask):
    
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


class TritonPythonModel:

    def initialize(self, args):
        current_name: str = str(Path(args["model_repository"]).parent.absolute())
        self.device = "cpu" if args["model_instance_kind"] == "CPU" else "cuda"
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')    
        self.scheduler_config_path = current_name + "/sam_stable_diffusion_inpaint/1/scheduler/"
        self.scheduler = PNDMScheduler.from_config(self.scheduler_config_path)

        self.height = 512
        self.width = 512
        self.vae_scale_factor = 8
        self.num_channels_latents = 4
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.eta = 0.0


    def prepare_latents(self, batch_size, dtype, generator, latents=None):
        shape = (batch_size, self.num_channels_latents, self.height // self.vae_scale_factor, self.width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = torch.randn(shape, generator=generator, device=self.device, dtype=dtype)
        else:
            latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, dtype, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(self.height // self.vae_scale_factor, self.width // self.vae_scale_factor)
        )
        mask = mask.to(device=self.device, dtype=dtype)

        masked_image = masked_image.to(device=self.device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        inputs = [pb_utils.Tensor.from_dlpack("sample", torch.to_dlpack(masked_image.contiguous()))]
        
        inference_request = pb_utils.InferenceRequest(
            model_name="vae_encoder",
            requested_output_names=["latent_sample"],
            inputs=inputs,
        )

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )
        else:
            output = pb_utils.get_output_tensor_by_name(
                inference_response, "latent_sample"
            )
            masked_image_latents: torch.Tensor = torch.from_dlpack(output.to_dlpack())
            masked_image_latents.generator = generator

            masked_image_latents = 0.18215 * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=self.device, dtype=dtype)
        return mask, masked_image_latents
        

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
        

    def execute(self, requests):
        responses = []

        for request in requests:
            prompt = [
                t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "PROMPT")
                .as_numpy()
                .tolist()
            ]

            negative_prompt = [
                t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "NEGATIVE_PROMPT")
                .as_numpy()
                .tolist()
            ]

            num_images_per_prompt = [
                t for t in pb_utils.get_input_tensor_by_name(request, "SAMPLES")
                .as_numpy()
                .tolist()
            ][0]

            self.num_inference_steps = [
                t for t in pb_utils.get_input_tensor_by_name(request, "STEPS")
                .as_numpy()
                .tolist()
            ][0]

            self.guidance_scale = [
                t for t in pb_utils.get_input_tensor_by_name(request, "GUIDANCE_SCALE")
                .as_numpy()
                .tolist()
            ][0]

            seed = [
                t for t in pb_utils.get_input_tensor_by_name(request, "SEED")
                .as_numpy()
                .tolist()
            ][0]
            
            pos_coords = pb_utils.get_input_tensor_by_name(request, "POS_COORDS").as_numpy()
            neg_coords = pb_utils.get_input_tensor_by_name(request, "NEG_COORDS").as_numpy()
            image = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy()
            image_size = image.shape[:2]

            # sam encoder
            image_origin = image.copy()
            image = prepare_image(image, ResizeLongestSide(1024), normalize=True)

            inputs = [pb_utils.Tensor.from_dlpack("inputs", torch.to_dlpack(image))]

            inference_request = pb_utils.InferenceRequest(
                model_name="sam_image_encoder",
                requested_output_names=["outputs"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "outputs"
                )
                image_embedding: torch.Tensor = torch.from_dlpack(output.to_dlpack())
            
            # sam
            input_point = np.concatenate([pos_coords, neg_coords])
            input_label = np.concatenate([np.ones(len(pos_coords)), np.zeros(len(neg_coords))])

            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]  # shape (1, -1, 2)
            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)  # shape(1, -1)

            resize_transform = ResizeLongestSide(1024)
            onnx_coord = resize_transform.apply_coords(onnx_coord, image_size).astype(np.float32)

            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)



            inputs = [
                pb_utils.Tensor.from_dlpack(
                    "image_embeddings", torch.to_dlpack(image_embedding)
                ),
                pb_utils.Tensor(
                    "point_coords", onnx_coord
                ),
                pb_utils.Tensor(
                    "point_labels", onnx_label
                ),
                pb_utils.Tensor(
                    "mask_input", onnx_mask_input
                ),
                pb_utils.Tensor(
                    "has_mask_input", onnx_has_mask_input
                ),
                pb_utils.Tensor(
                    "orig_im_size", np.array(image_size, dtype=np.float32)
                )
            ]

            inference_request = pb_utils.InferenceRequest(
                model_name="sam",
                requested_output_names=["masks", "iou_predictions", "low_res_masks"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:

                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "masks"
                )
                mask = torch.from_dlpack(output.to_dlpack()).cpu().numpy().squeeze()


            # stable diffusion inpaint
            if negative_prompt[0] == "NONE":
                negative_prompt = None


            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_input_ids = text_input.input_ids
            
            input_ids = text_input_ids.type(dtype=torch.int32)
            inputs = [
                pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))
            ]

            inference_response = pb_utils.InferenceRequest(
                model_name="text_encoder",
                requested_output_names=["last_hidden_state"],
                inputs=inputs,
            ).exec()

            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "last_hidden_state"
                )
                text_embeddings: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
            text_embeddings = text_embeddings.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )

            do_classifier_free_guidance = self.guidance_scale > 1.0
            batch_size = 1

            if do_classifier_free_guidance:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt


                max_length = text_input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = uncond_input.input_ids.type(dtype=torch.int32)
                inputs = [
                    pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids))
                ]

                inference_request = pb_utils.InferenceRequest(
                    model_name="text_encoder",
                    requested_output_names=["last_hidden_state"],
                    inputs=inputs,
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    output = pb_utils.get_output_tensor_by_name(
                        inference_response, "last_hidden_state"
                    )
                    uncond_embeddings: torch.Tensor = torch.from_dlpack(
                        output.to_dlpack()
                    )

                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = uncond_embeddings.shape[1]
                uncond_embeddings = uncond_embeddings.repeat(
                    1, num_images_per_prompt, 1
                )
                uncond_embeddings = uncond_embeddings.view(
                    batch_size * num_images_per_prompt, seq_len, -1
                )

                text_embeddings = torch.cat([uncond_embeddings.to(self.device), text_embeddings.to(self.device)])


            image = prepare_image(image_origin, ResizeLongestSide(512)).transpose(1, 2, 0)
            mask = prepare_image(mask, ResizeLongestSide(512))

            mask, masked_image = prepare_mask_and_masked_image(image, mask)

            self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            generator = torch.Generator(device=self.device).manual_seed(seed)

            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                text_embeddings.dtype,
                generator
            )

            mask, masked_image_latents = self.prepare_mask_latents(
                mask,
                masked_image,
                batch_size * num_images_per_prompt,
                torch.float16,
                generator,
                do_classifier_free_guidance,
            )

            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]

            
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.eta)

            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                latent_model_input = latent_model_input.type(dtype=torch.float16)
                timestep = t[None].type(dtype=torch.float16)
                encoder_hidden_states = text_embeddings.type(dtype=torch.float16)

                inputs = [
                    pb_utils.Tensor.from_dlpack(
                        "sample", torch.to_dlpack(latent_model_input)
                    ),
                    pb_utils.Tensor.from_dlpack("timestep", torch.to_dlpack(timestep)),
                    pb_utils.Tensor.from_dlpack(
                        "encoder_hidden_states", torch.to_dlpack(encoder_hidden_states)
                    ),
                ]

                inference_request = pb_utils.InferenceRequest(
                    model_name="unet",
                    requested_output_names=["out_sample"],
                    inputs=inputs,
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                else:
                    output = pb_utils.get_output_tensor_by_name(
                        inference_response, "out_sample"
                    )
                    noise_pred: torch.Tensor = torch.from_dlpack(output.to_dlpack())

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred.to(self.device), t.to(self.device), latents.to(self.device), **extra_step_kwargs).prev_sample

            latents = 1 / 0.18215 * latents

            latents = latents.type(dtype=torch.float16)
            inputs = [pb_utils.Tensor.from_dlpack("latent_sample", torch.to_dlpack(latents))]
            inference_request = pb_utils.InferenceRequest(
                model_name="vae_decoder",
                requested_output_names=["sample"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "sample"
                )
                image: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                image = image.type(dtype=torch.float32)
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()

            tensor_output = [pb_utils.Tensor("IMAGES", image)]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses