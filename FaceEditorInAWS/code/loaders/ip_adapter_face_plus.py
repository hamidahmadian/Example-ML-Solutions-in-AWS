from typing import Dict, List, Union

import torch
from diffusers.utils.hub_utils import _get_model_file
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..helper.nn_modules import ProjPlusModel, LoRAAttnProcessor, LoRAIPAttnProcessor


class IpAdapterFacePlusMixin:
    _image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    num_tokens = 4
    lora_rank = 128

    def set_encoder_hid_proj(self, state_dict):
        self.image_proj_model = ProjPlusModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=self.dtype)

        self.image_proj_model.load_state_dict(state_dict["image_proj"])

    def set_ip_adapter(self, state_dict):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank,
                ).to(self.device, dtype=self.dtype)
            else:
                attn_procs[name] = LoRAIPAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=self.lora_rank,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.dtype)
        unet.set_attn_processor(attn_procs)

        ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    def load_ip_adapter_face_plus(
            self,
            pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, torch.Tensor]],
            weight_name: Union[str, List[str]],
            subfolder: Union[str, List[str]] = None,
            **kwargs,
    ):
        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["image_proj", "ip_adapter"]:
            raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

        # if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self._image_encoder_path).to(
            self.device, dtype=self.dtype
        )

        self.clip_image_processor = CLIPImageProcessor()

        self.set_encoder_hid_proj(state_dict)
        self.set_ip_adapter(state_dict)

    def set_ip_adapter_face_plus_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.scale = scale
