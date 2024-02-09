import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKL
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from pipelines.diffusion_pipelines import IpAdapterFacePlusPipeline


def model_fn(model_dir):
    # Model Path, Load models and move to GPU
    base_model_path = f"{model_dir}/SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = f"{model_dir}/stabilityai/sd-vae-ft-mse"
    image_encoder_path = f"{model_dir}/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = f"{model_dir}/ip-adapter-faceid-plus_sd15.bin"
    device = "cuda"

    # Load scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # Load vae
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    pipe = IpAdapterFacePlusPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )

    # Move pipe to cuda
    pipe = pipe.to(device)

    # Load adapter
    pipe.load_ip_adapter_face_plus(pretrained_model_name_or_path_or_dict=ip_ckpt,
                                   weight_name='ip-adapter-faceid-plus_sd15.bin',
                                   image_encoder_path=image_encoder_path)

    return pipe


def predict_fn(data, pipe):
    # Get image and Load faceid embeds and face image

    # Load PIL image
    buffer_image = BytesIO(base64.b64decode(data.pop("image")))
    pil_image = Image.open(buffer_image).convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces = app.get(open_cv_image)

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(open_cv_image, landmark=faces[0].kps,
                                      image_size=224)  # you can also segment the face

    # get prompt & parameters
    v2 = False
    prompt = data.pop("prompt", data)
    # set valid HP for stable diffusion
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
    num_inference_steps = data.pop("num_inference_steps", 30)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)

    # run generation with parameters
    generated_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        face_image=face_image,
        faceid_embeds=faceid_embeds,
        shortcut=v2,
        scale=1.0,
        s_scale=1.0,
        num_samples=num_images_per_prompt,
        width=512,
        height=768,
        num_inference_steps=num_inference_steps
    ).images

    # create response
    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # create response
    return {"generated_images": encoded_images}
