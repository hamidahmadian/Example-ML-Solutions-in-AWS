from typing import List

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from ..loaders.ip_adapter_face_plus import IpAdapterFacePlusMixin


class IpAdapterFacePlusPipeline(StableDiffusionPipeline, IpAdapterFacePlusMixin):

    @torch.inference_mode()
    def get_image_embeds(self, faceid_embeds, face_image, s_scale, shortcut):
        if isinstance(face_image, Image.Image):
            pil_image = [face_image]
        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]

        faceid_embeds = faceid_embeds.to(self.device, dtype=self.dtype)
        image_prompt_embeds = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds,
                                                           shortcut=shortcut, scale=s_scale)
        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    @torch.no_grad()
    def __call__(self,
                 face_image=None,
                 faceid_embeds=None,
                 prompt=None,
                 negative_prompt=None,
                 scale=1.0,
                 num_samples=4,
                 seed=None,
                 guidance_scale=7.5,
                 num_inference_steps=30,
                 s_scale=1.0,
                 shortcut=False,
                 **kwargs):

        self.set_ip_adapter_face_plus_scale(scale)

        num_prompts = faceid_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds, face_image, s_scale,
                                                                                shortcut)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        print(prompt_embeds.shape)
        print(kwargs)
        return super().__call__(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        )
