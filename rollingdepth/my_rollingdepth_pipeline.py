# Copyright 2024 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-11-29
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/RollingDepth#-citation
# More information about the method can be found at https://rollingdepth.github.io
# ---------------------------------------------------------------------------------

import logging
from os import PathLike
from typing import Dict, List, Union
from PIL import Image
import numpy as np

import einops
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,  # type: ignore
    DDIMScheduler,  # type: ignore
    DiffusionPipeline,  # type: ignore
    UNet2DConditionModel,  # type: ignore
)
from diffusers.utils import BaseOutput  # type: ignore
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

from .depth_aligner import DepthAligner
from .video_io import load_video_frames
import os
import numpy as np
from PIL import Image


import cv2


import os
import math
import numpy as np
import torch
import safetensors.torch as sf
#import db_examples

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file


sd15_name = 'models/stablediffusionapi-realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("models/models--briaai--RMBG-1.4")

# with torch.no_grad():
#     new_conv_in = torch.nn.Conv2d(12, self.unet.conv_in.out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding,dtype=torch.float16)
#     new_conv_in.weight.zero_()
#     new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)

#     new_conv_in.bias = self.unet.conv_in.bias
#     self.unet.conv_in = new_conv_in


# def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
#     c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
#     c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
#     new_sample = torch.cat([sample, c_concat], dim=1)
#     kwargs['cross_attention_kwargs'] = {}
#     return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


# unet.forward = hooked_unet_forward

# # Load

# model_path = './models/iclight_sd15_fbc.safetensors'

# if not os.path.exists(model_path):
#     download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=model_path)

# sd_offset = sf.load_file(model_path)
# sd_origin = unet.state_dict()
# keys = sd_origin.keys()
# sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
# unet.load_state_dict(sd_merged, strict=True)
# del sd_offset, sd_origin, sd_merged, keys

# # Device


# text_encoder = text_encoder.to(device=device, dtype=torch.float16)
# vae = vae.to(device=device, dtype=torch.bfloat16)
# unet = unet.to(device=device, dtype=torch.float16)
# rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

def resize_frames(input_frames, new_width, new_height):
    """
    对输入的图像序列进行调整大小操作
    :param input_frames: 输入的图像序列，类型为torch.Tensor，形状可能为[batch_size, channels, height, width]等
    :param new_width: 目标宽度
    :param new_height: 目标高度
    :return: 调整大小后的图像序列，类型仍为torch.Tensor
    """
    resized_frames = []
    # 将torch.Tensor转换为numpy.ndarray类型并调整大小，再转换回torch.Tensor
    for frame in input_frames:
        frame_np = frame.detach().cpu().numpy()
        if len(frame_np.shape) == 3:  # 例如彩色图像 (height, width, channels)
            frame_np = np.transpose(frame_np, (1, 2, 0))  # 转换为 (width, height, channels) 以符合cv2的要求
            resized_np = cv2.resize(frame_np, (new_width, new_height))
            resized_np = np.transpose(resized_np, (2, 0, 1))  # 再转换回 (channels, width, height)
            resized_frames.append(torch.from_numpy(resized_np).to(frame.device))
        elif len(frame_np.shape) == 2:  # 例如灰度图像 (height, width)
            resized_np = cv2.resize(frame_np, (new_width, new_height))
            resized_frames.append(torch.from_numpy(resized_np).to(frame.device))
    return torch.stack(resized_frames)

@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]
    
    device = torch.device('cpu')
    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)





@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha

class RollingDepthOutput(BaseOutput):
    input_rgb: torch.Tensor
    depth_pred: torch.Tensor
    R_pred : torch.Tensor
    G_pred : torch.Tensor
    B_pred : torch.Tensor
    aligned_snippet_pred_ls: Union[None, List[torch.Tensor]]
    # intermediate results
    snippet_ls: Union[None, List[torch.Tensor]]
    depth_coaligned: Union[None, torch.Tensor]

class RollingDepthPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    # rgb_latent_scale_factor = vae.config.scaling_factor
    # depth_latent_scale_factor = vae.config.scaling_factor
    N_CHANNEL_PER_LATENT = 4

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        #scheduler: DDIMScheduler,
        scheduler: DPMSolverMultistepScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=dpmpp_2m_sde_karras_scheduler,
            #dpmpp_2m_sde_karras_scheduler=dpmpp_2m_sde_karras_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed: torch.Tensor = None  # type: ignore

        logging.debug(f"Pipeline initialized: {type(self)}")

    @torch.no_grad()
    def __call__(
        self,
        # input setting
        input_fg_video_path: PathLike,
        input_bg_video_path: PathLike,
        start_frame: int = 0,
        frame_count: int = 253,
        processing_res: int = 1024,
        resample_method: str = "BILINEAR",
        # infer setting
        dilations: List[int] = [1, 25],
        cap_dilation: bool = True,
        snippet_lengths: List[int] = [3],
        init_infer_steps: List[int] = [1],
        strides: List[int] = [1],
        coalign_kwargs: Union[Dict, None] = None,
        refine_step: int = 0,
        refine_snippet_len: int = 3,
        refine_start_dilation: int = 6,
        # other settings
        generator: Union[torch.Generator, None] = None,
        verbose: bool = False,
        max_vae_bs: int = 4,
        unload_snippet: bool = False,
        # output settings
        restore_res: bool = False,
    ) -> RollingDepthOutput:
        assert processing_res >= 0

        # ----------------- Check settings -----------------
        if processing_res > 1024:
            logging.warning(
                f"Procssing at high-resolution ({processing_res}) may lead to suboptimal accuracy."
            )

        # ----------------- Load input data -----------------
        # Load, resize, and normalize input frames
        seed=12345
        rng = torch.Generator(device=self.device).manual_seed(seed)


        new_width = 720
        new_height = 640
        input_fg_frames, original_res_fg = load_video_frames(
            input_path=input_fg_video_path,
            start_frame=start_frame,
            frame_count=253,
            processing_res=processing_res,
            resample_method=resample_method,
            verbose=verbose,
        )
        # 对前景图像序列进行resize
        input_fg_frames = resize_frames(input_fg_frames, new_width, new_height)
        input_fg_frames = einops.rearrange(input_fg_frames, "n c h w -> 1 n c h w")
        logging.info(
            f"{input_fg_frames.shape[1]} frames loaded from video {input_fg_video_path}"
        )
        input_bg_frames, original_res_bg = load_video_frames(
            input_path=input_bg_video_path,
            start_frame=start_frame,
            frame_count=frame_count,
            processing_res=processing_res,
            resample_method=resample_method,
            verbose=verbose,
        )
        # 对背景图像序列进行resize
        input_bg_frames = resize_frames(input_bg_frames, new_width, new_height)
        input_bg_frames = einops.rearrange(input_bg_frames, "n c h w -> 1 n c h w")
        logging.info(
            f"{input_bg_frames.shape[1]} frames loaded from video {input_bg_video_path}"
        )

        # warn if resize back to big resolution
        if restore_res and max(original_res) > 2048:  # type: ignore
            logging.warning(
                f"Resizing back to large resolution ({list(original_res)}) may result in significant memory usage."  # type: ignore
            )

        # ----------------- Predicting depth -----------------
        pipe_output = self.forward(
            input_fg_frames=input_fg_frames,
            input_bg_frames=input_bg_frames,
            dilations=dilations,
            cap_dilation=cap_dilation,
            snippet_lengths=snippet_lengths,
            init_infer_steps=init_infer_steps,
            strides=strides,
            coalign_kwargs=coalign_kwargs,
            refine_step=refine_step,
            refine_snippet_len=refine_snippet_len,
            refine_start_dilation=refine_start_dilation,
            verbose=verbose,
            generator=rng,
            max_vae_bs=max_vae_bs,
            unload_snippet=unload_snippet,
        )

        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Resize back -----------------
        if restore_res:
            if verbose:
                logging.info(f"Resizing to the original resolution: {original_res}")
            # Restore RGB resolution
            input_rgb = pipe_output.input_rgb
            input_rgb = resize(
                input_rgb,
                list(original_res),
                interpolation=InterpolationMode.__getitem__(resample_method),
            )
            pipe_output.input_rgb = input_rgb

            # Restore depth resolution
            depth_pred = pipe_output.depth_pred
            depth_pred = resize(
                depth_pred,
                list(original_res),
                interpolation=InterpolationMode.__getitem__(resample_method),
            )
            pipe_output.depth_pred = depth_pred

        return pipe_output

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
    
    


    #@torch.inference_mode()
    def run_rmbg(img, sigma=0.0):
        H, W, C = img.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
        alpha = rmbg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
        return result.clip(0, 255).astype(np.uint8), alpha
    

    @torch.no_grad()
    def forward(
        self,
        input_fg_frames: torch.Tensor,  # [1 N 3 H W]
        input_bg_frames: torch.Tensor,  # [1 N 3 H W]
        # infer setting
        dilations: List[int],
        cap_dilation: bool,
        snippet_lengths: List[int],
        init_infer_steps: List[int],
        strides: List[int],
        coalign_kwargs: Union[Dict, None],
        refine_step: int,
        refine_snippet_len: int,
        refine_start_dilation: int,
        # other setting
        generator: Union[torch.Generator, None],
        verbose: bool,
        max_vae_bs: int,
        unload_snippet: bool,
    ) -> RollingDepthOutput:
        # ----------------- Check settings -----------------
        assert 1 in dilations, "dilations should include 1"
        # snippet_length
        assert len(snippet_lengths) == len(
            set(snippet_lengths)
        ), f"Repeated values found in {snippet_lengths = }"
        if len(snippet_lengths) > 1:
            assert (
                len(snippet_lengths) == len(dilations)
            ), f"inconsistent lengths of window_size ({snippet_lengths}) and dilations ({dilations})"
        else:
            snippet_lengths = snippet_lengths * len(dilations)
        # infer denoise steps
        if len(init_infer_steps) > 1:
            assert (
                len(init_infer_steps) == len(dilations)
            ), f"inconsistent lengths of init_infer_step ({init_infer_steps}) and dilations ({dilations})"
        else:
            init_infer_steps = init_infer_steps * len(dilations)
        assert min(init_infer_steps) > 0, "Minimum inference step is 1"
        # stride
        if len(strides) > 1:
            assert (
                len(strides) == len(dilations)
            ), f"inconsistent lengths of strides ({strides}) and dilations ({dilations})"
        else:
            strides = strides * len(dilations)
        if [1] * len(dilations) != strides:
            raise NotImplementedError("Only implemented for stride 1")

        # Cap dilation
        seq_len = input_fg_frames.shape[1]
        if cap_dilation:
            for i, dilation in enumerate(dilations):
                dilations[i] = self.cap_max_dilation(
                    seq_len=seq_len,
                    snippet_len=snippet_lengths[i],
                    dilation=dilation,
                    verbose=verbose,
                )
            refine_start_dilation = self.cap_max_dilation(
                seq_len=seq_len,
                snippet_len=refine_snippet_len,
                dilation=refine_start_dilation,
                verbose=verbose,
            )

        # ----------------- Initial prediction -----------------
        device = self.device

        seed=12345
        rng = torch.Generator(device=self.device).manual_seed(seed)
        
        device = torch.device('cuda')
        # self.text_encoder = text_encoder.to(device=device, dtype=torch.float16)
        self.vae = vae.to(device=device, dtype=torch.float16)
        self.unet = unet.to(device=device, dtype=torch.float16)

        input_fg_frames = input_fg_frames.to(self.dtype).to(device)
        input_bg_frames = input_bg_frames.to(self.dtype).to(device)
        
        print(input_fg_frames.shape)
        print(input_bg_frames.shape)
        print("***************************")

        #print(input_bg_frames[].shape)

        # 用于存储合并后的结果
        concat_conds = []

        # 遍历每一对对应的前景和背景帧
        for i in range(input_fg_frames.shape[0]):
            fg_frame = input_fg_frames[i].unsqueeze(0)  # 增加一个维度，使其符合后续拼接要求（模拟单张图像的批量维度）
            bg_frame = input_bg_frames[i].unsqueeze(0)
            print(fg_frame.shape)
            print(bg_frame.shape)
            combined_frames = torch.cat([fg_frame, bg_frame], dim=0)  # 在通道维度拼接
            print("combined_frames.shape",combined_frames.shape)
            concat_conds.append(combined_frames)

        concat_conds = torch.cat(concat_conds, dim=0).to(device=vae.device, dtype=vae.dtype)
        print("concat_conds.shape",concat_conds.shape)
        # 将结果列表转换为张量并移动到指定设备、设置为指定数据类型
        #concat_conds = torch.cat(concat_conds, dim=0).to(device=vae.device, dtype=vae.dtype)
        # Encode RGB frames
        # rgb_latent_fg = self.encode_rgb(
        #     input_fg_frames, max_batch_size=max_vae_bs, verbose=verbose
        # )
        # rgb_latent_bg = self.encode_rgb(
        #     input_bg_frames, max_batch_size=max_vae_bs, verbose=verbose
        # )

        # print(rgb_latent_fg.shape)
        # print(rgb_latent_bg.shape)


        # 这里直接对对应的fg_bg用process？？  是不是就和我之前那种没区别了。
        # 现在就是我需要把iclight的denoise过程怎么放到marigold里
        #rgb_latent_bg = rgb_latent_bg.permute(0, 1, 2, 4, 3).contiguous()

        # 进行拼接
        # concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
        #concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)
        
        concat_conds = concat_conds.to(torch.float16)
        rgb_latent = self.encode_rgb(
            concat_conds, max_batch_size=max_vae_bs, verbose=verbose
        )
        print("rgb_latent.shape before cat c",rgb_latent.shape)
        rgb_latent = torch.cat([c[None, ...] for c in rgb_latent], dim=2)
        print("rgb_latent.shape",rgb_latent.shape)

        #rgb_latent=concat_conds
        #rgb_latent = torch.cat([rgb_latent_fg, rgb_latent_bg], dim=2)

        # Empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()

        B, seq_len, _, h, w = rgb_latent.shape

        if 1 != B:
            raise NotImplementedError("Layered inference is only implemented for B=1")

        torch.cuda.empty_cache()

        # Use the same for every frame
        init_noise = torch.randn(
            (B, 4, h, w),
            device=device,
            dtype=self.dtype,
            generator=rng,
        )
        init_noise = einops.repeat(init_noise, "1 c h w -> B n c h w", n=seq_len, B=B)

        # Get snippets
        snippet_pred_ls = self.init_snippet_infer(
            rgb_latent=rgb_latent,
            init_noise=init_noise,
            dilations=dilations,
            cap_dilation=cap_dilation,
            snippet_lengths=snippet_lengths,
            init_infer_steps=init_infer_steps,
            strides=strides,
            max_vae_bs=max_vae_bs,
            unload_snippet=unload_snippet,
            verbose=verbose,
        )
        print("snippet_pred_ls 的 shape before alignment",snippet_pred_ls[0].shape)
        print("snippet_pred_ls 的 length before alignment",snippet_lengths)


        # 假设这是你的数据，这里直接用你提供的形状示例来模拟一个随机张量，实际中替换为真实的 snippet_pred_ls[0]
        snippet_pred = snippet_pred_ls[0]

        # 创建一个列表用于存储图像数据（调整格式后）
        image_list = []
        for i in range(snippet_pred.shape[0]):
            image = snippet_pred[i, 0].permute(1, 2, 0).cpu().numpy()  # 将通道维度调整到最后，并转换为 numpy 数组，方便 imageio 使用
            image_list.append(image)

        cv2.imwrite("img0 before alignment.jpg",image_list[0])
        cv2.imwrite("img50 before alignment.jpg",image_list[50])

        # # 使用 imageio 保存为 gif 文件，你可以根据喜好选择其他格式，如 'mp4' 等
        # imageio.mimsave('snippet_pred_images.gif', image_list, fps=5)  # fps 表示每秒显示的帧数，可根据需要调整
        
        # ----------------- Co-alignment -----------------
        coalign_kwargs = {} if coalign_kwargs is None else coalign_kwargs
        depth_aligner = DepthAligner(
            verbose=verbose,
            device=device,
            **coalign_kwargs,
        )
         ########################################
        #对每个通道单独对齐

        aligned_snippet_pred_ls = []  # 保存对齐后的片段

                # 定义存储每个通道的列表
        r_list = []
        g_list = []
        b_list = []

        # 遍历 snippet_pred_ls 中的每个 triplet
        for triplets in snippet_pred_ls:
            # 打印 triplets 的形状，确保它是 [B, N, 3, H, W]
            print(triplets.shape)

            # 分离为 R、G、B 通道 [B, N, 1, H, W]
            r_channel, g_channel, b_channel = triplets.split(1, dim=2)  
            print(r_channel.shape)
            print(g_channel.shape)
            print(b_channel.shape)

            # 将每个通道添加到对应的列表中
            r_list.append(r_channel)
            g_list.append(g_channel)
            b_list.append(b_channel)
            # 分别对 R、G、B 通道进行 align 操作
        (R_coaligned, scales, translations, loss_history) = depth_aligner.run(
        snippet_ls=r_list, dilations=dilations
    )
        # Re-normalize
        R_coaligned -= R_coaligned.min()
        R_coaligned /= R_coaligned.max()
        R_coaligned = R_coaligned * 2.0 - 1.0

        (G_coaligned, scales, translations, loss_history) = depth_aligner.run(
        snippet_ls=g_list, dilations=dilations
    )
        # Re-normalize
        G_coaligned -= G_coaligned.min()
        G_coaligned /= G_coaligned.max()
        G_coaligned = G_coaligned * 2.0 - 1.0

        (B_coaligned, scales, translations, loss_history) = depth_aligner.run(
        snippet_ls=b_list, dilations=dilations
    )
        # Re-normalize
        B_coaligned -= B_coaligned.min()
        B_coaligned /= B_coaligned.max()
        B_coaligned = B_coaligned * 2.0 - 1.0
        # # 将对齐后的通道合并回 RGB 图像
        aligned_triplets = torch.cat([R_coaligned, G_coaligned, B_coaligned], dim=2)  # [B, N, 3, H, W]

        # # 保存到新的列表
        aligned_snippet_pred_ls.append(aligned_triplets)

        # # 返回对齐后的片段列表
        # return aligned_snippet_pred_ls

        #(depth_coaligned, scales, translations, loss_history) = depth_aligner.run(
        #     snippet_ls=snippet_pred_ls, dilations=dilations
        # )

        # # Re-normalize
        # depth_coaligned -= depth_coaligned.min()
        # depth_coaligned /= depth_coaligned.max()
        # depth_coaligned = depth_coaligned * 2.0 - 1.0

        torch.cuda.empty_cache()

        # ----------------- Refinement -----------------
        if refine_step > 0:
            # Encode depth
            depth_latent_coaligned = self.encode_rgb(
                einops.repeat(depth_coaligned, "N 1 H W -> 1 N 3 H W"),
                max_batch_size=max_vae_bs,
                verbose=verbose,
            )
            # Refine
            depth_latent_new = self.refine(
                rgb_latent=rgb_latent,
                depth_latents=depth_latent_coaligned,
                init_noise=init_noise,
                refine_step=refine_step,
                snippet_len=refine_snippet_len,
                start_dilation=refine_start_dilation,
                verbose=verbose,
            )
            # Decode
            depth_pred = self.decode_depth(
                depth_latent_new, max_batch_size=max_vae_bs, verbose=verbose
            )
        else:
            #depth_pred = depth_coaligned
            R_pred = R_coaligned
            G_pred = G_coaligned
            B_pred = B_coaligned

        # ----------------- Output -----------------
        pipe_out = RollingDepthOutput(
            #input_rgb=input_frames.detach().cpu().squeeze(0) / 2.0 + 0.5,
            #depth_pred=depth_pred.detach().cpu().squeeze(0),
            R_pred=R_pred.detach().cpu().squeeze(0)/ 2.0 + 0.5,
            G_pred=G_pred.detach().cpu().squeeze(0)/ 2.0 + 0.5,
            B_pred=B_pred.detach().cpu().squeeze(0)/ 2.0 + 0.5,
            # R_pred=R_pred.detach().cpu().squeeze(0),
            # G_pred=G_pred.detach().cpu().squeeze(0),
            # B_pred=B_pred.detach().cpu().squeeze(0),
            snippet_ls=[snippet.detach().cpu() for snippet in snippet_pred_ls],
            #depth_coaligned=depth_coaligned.detach().cpu().squeeze(0),
            aligned_snippet_pred_ls=[aligned_triplets.detach().cpu()for aligned_triplets in aligned_snippet_pred_ls],
        )
        return pipe_out


    def init_snippet_infer(
        self,
        rgb_latent: torch.Tensor,
        init_noise: torch.Tensor,
        dilations: List[int],
        cap_dilation: bool,
        snippet_lengths: List[int],
        init_infer_steps: List[int],
        strides: List[int],
        max_vae_bs: int,
        unload_snippet: bool,
        verbose: bool,
    ) -> List[torch.Tensor]:
        device = self.device

        B, seq_len, _, h, w = rgb_latent.shape  # latent shape
        if 1 != B:
            raise NotImplementedError("RollingDepth is implemented for B=1")

        # Empty text embedding
        batch_empty_text_embed = self.empty_text_embed.to(device)  # [1, 2, 1024]

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(12, self.unet.conv_in.out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding,dtype=torch.float16)
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)

            new_conv_in.bias = self.unet.conv_in.bias
            self.unet.conv_in = new_conv_in
        # Load

        model_path = './models/iclight_sd15_fbc.safetensors'

        if not os.path.exists(model_path):
            download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=model_path)

        sd_offset = sf.load_file(model_path)
        sd_origin = self.unet.state_dict()
        keys = sd_origin.keys()
        #sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        sd_merged = {k: sd_origin[k].to('cuda') + sd_offset[k].to('cuda') for k in sd_origin.keys()}
        self.unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged, keys
        
        
        device = torch.device('cuda')
        self.text_encoder = text_encoder.to(device=device, dtype=torch.float16)
        self.vae = vae.to(device=device, dtype=torch.float16)
        self.unet = unet.to(device=device, dtype=torch.float16)
        # # rmbg = rmbg.to(device=device, dtype=torch.float32)
        # Output
        snippet_pred_ls = []

        # >>> Go through dilations >>>
        iterable_init_infer = zip(dilations, snippet_lengths, strides, init_infer_steps)
        if verbose:
            iterable_init_infer = tqdm(
                iterable_init_infer,
                desc=" Initial snippet inference",
                leave=False,
                total=len(dilations),
            )
        for dilation, snippet_len, stride, init_infer_step in iterable_init_infer:
            # Set timesteps
            self.scheduler.set_timesteps(init_infer_step, device=device)
            timesteps = self.scheduler.timesteps  # [T]

            # Indice of snippet frames
            snippet_idx_ls = self.get_snippet_indice(
                i_step=0,
                timesteps=timesteps,
                seq_len=seq_len,
                snippet_len=snippet_len,
                dilation_start=dilation,
                dilation_end=dilation,
                stride=stride,
            )
            
            # >> Go through snippets >>
            depth_snippet_latent_ls = []
            #********************************
            rgb_snippet_latent_ls = []
            #********************************
            snippet_iterable = snippet_idx_ls
            if verbose:
                snippet_iterable = tqdm(
                    snippet_iterable,
                    desc=f"{' '*2}Predicting snippets with dilation {dilation}",
                    leave=False,
                )
            for snippet_idx in snippet_iterable:
                
                # rng = torch.Generator(device=device).manual_seed(12345)
                # prompt = "Pouring water into a glass , natural lighting"
                # a_prompt = "best quality"
                # n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
                # conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)
                # device = torch.device('cuda')
                # self.text_encoder = self.text_encoder.to(device=device, dtype=torch.float16)
                # self.vae = self.vae.to(device=device, dtype=torch.bfloat16)
                # self.unet = self.unet.to(device=device, dtype=torch.float16)
                # self.rmbg = self.rmbg.to(device=device, dtype=torch.float32)
                # Get input frames
                rgb_latent_snippet = rgb_latent[:, snippet_idx, :, :, :].clone()
                depth_latent_snippet = init_noise[:, snippet_idx, :, :, :].clone()
                #depth_latent_snippet=depth_latent_snippet.to(torch.float16)
                # depth_latent_snippet = depth_latent_snippet.to(torch.float32)
                # self.unet = self.unet.to(device=device, dtype=torch.float16)
                #new_tensor = tensor[:, 0, :, :, :]
                # triplets_decoded=pixels
                #pixels = pytorch2numpy(pixels, quant=False)
                
                # return pixels, [fg, bg]
                # Reset timesteps
                self.scheduler.set_timesteps(init_infer_step, device=device)
                timesteps = self.scheduler.timesteps  # [T]

                # Denoising loop
                iterable_step = timesteps
                if verbose and init_infer_step > 1:
                    iterable_step = tqdm(
                        iterable_step,
                        total=len(timesteps),
                        leave=False,
                        desc=f"{' '*3}Denoising",
                    )
                for t_current in iterable_step:
                    t_input = t_current.repeat(rgb_latent_snippet.shape[1])
                    # Denoising step
                    noise_pred = self.single_step(
                        rgb_latent=rgb_latent_snippet,
                        depth_latent=depth_latent_snippet,
                        timestep=t_input,
                        encoder_hidden_states=batch_empty_text_embed,
                    )
                    _scheduler_output = self.scheduler.step(
                        noise_pred, t_current, depth_latent_snippet
                    )
                    # _scheduler_output = self.scheduler.step(
                    #     noise_pred, t_current, rgb_latent_snippet
                    # )
                    depth_latent_snippet = _scheduler_output.prev_sample
                    #rgb_latent_snippet = _scheduler_output.prev_sample
                depth_snippet_latent_ls.append(depth_latent_snippet)
                rgb_snippet_latent_ls.append(rgb_latent_snippet)
                #rgb_snippet_latent_ls.append(rgb_latent_snippet)
            # << Go through snippets <<
            depth_snippet_latent = torch.concat(depth_snippet_latent_ls, dim=0)
            # rgb_snippet_latent = torch.concat(rgb_snippet_latent_ls, dim=0)

            # Decode to depth
            del depth_snippet_latent_ls
            triplets_decoded = self.decode_depth(
                depth_snippet_latent, max_batch_size=max_vae_bs, verbose=verbose
            )

            # # Decode to rgb_img
            # del rgb_snippet_latent_ls
            # triplets_decoded = self.decode_depth(
            #     rgb_snippet_latent, max_batch_size=max_vae_bs, verbose=verbose
            # )
            # triplets_decoded=depth_snippet_latent
            # moved to CPU to save vram
            if unload_snippet:
                triplets_decoded = triplets_decoded.cpu()

            snippet_pred_ls.append(triplets_decoded)
            torch.cuda.empty_cache()
        # <<< Go through dilations <<<
        return snippet_pred_ls

    @staticmethod
    def get_snippet_indice(
        i_step: int,
        timesteps: torch.Tensor,
        seq_len: int,
        snippet_len: int,
        dilation_start: int,
        dilation_end: int,
        stride: int,
    ) -> List[List[int]]:
        gap_start = dilation_start - 1
        gap_end = dilation_end - 1
        assert (
            gap_start >= gap_end
        ), f"expect gap_start > gap_end, but got {gap_start} and {gap_end}"
        assert gap_start >= 0 and gap_end >= 0

        total_step = len(timesteps)
        gap_cur = int((1 - (i_step) / total_step) * (gap_start - gap_end) + gap_end)

        # Generate snippet indice
        snippet_idx_ls = []
        total_window_size = (snippet_len - 1) * (gap_cur + 1) + 1
        # index of the first frame
        i_start_ls = list(range(0, seq_len - total_window_size + 1, stride))
        # last window (for stride > 1)
        if i_start_ls[-1] < seq_len - total_window_size:
            i_start_ls.append(seq_len - total_window_size)
        for i_start in i_start_ls:
            input_idx = list(range(i_start, i_start + total_window_size, gap_cur + 1))
            snippet_idx_ls.append(input_idx)

        # Check if every frame is covered
        if not set(range(0, seq_len)) == set([x for f in snippet_idx_ls for x in f]):
            logging.warning(
                "Not every frame is covered. Consider reducing dilation for short videos"
            )
        return snippet_idx_ls

    @staticmethod
    def cap_max_dilation(seq_len: int, snippet_len: int, dilation: int, verbose: bool):
        # Cap by sequence_len
        max_allowed_gap = int(seq_len / snippet_len) - 1
        if max_allowed_gap < dilation:
            temp_msg = f"{dilation = } is too big for {seq_len} frames. Reduced to {max_allowed_gap}"
            if verbose:
                logging.info(temp_msg)
            else:
                logging.debug(temp_msg)
            dilation = min(max_allowed_gap, dilation)
        return dilation

    def refine(
        self,
        rgb_latent: torch.Tensor,
        depth_latents: torch.Tensor,
        init_noise: torch.Tensor,
        refine_step: int,
        snippet_len: int,
        start_dilation: int,
        verbose: bool,
        skip_t_ratio: float = 0.5,
    ) -> torch.Tensor:
        device = self.device
        B, seq_len, _, h, w = rgb_latent.shape  # latent shape

        if 1 != B:
            raise NotImplementedError("Layered inference is only implemented for B=1")

        # Set timesteps
        total_scheduler_step = int(refine_step / skip_t_ratio)
        assert total_scheduler_step <= self.scheduler.config.get(
            "num_train_timesteps"
        ), "Too many refinement steps"
        self.scheduler.set_timesteps(total_scheduler_step, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Start from intermediate step
        _start_idx = int(len(timesteps) * skip_t_ratio)
        timesteps = timesteps[_start_idx:]
        assert 0 < len(timesteps) < total_scheduler_step, f"invalid {skip_t_ratio = }"

        # Add noise to latent
        depth_latent_new = self.scheduler.add_noise(
            original_samples=depth_latents,
            noise=init_noise.clone().to(device, dtype=self.dtype),
            timesteps=timesteps[0],
        )

        # Empty text embedding
        batch_empty_text_embed = self.empty_text_embed.to(device)  # [1, 2, 1024]

        # Timestep of each frame
        frame_timestep = (
            torch.zeros((seq_len), device=device, dtype=timesteps.dtype) + timesteps[0]
        )

        # Denoising loop
        iterable_step = enumerate(timesteps)
        if verbose:
            iterable_step = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 2 + "Diffusion denoising",
            )
        for i_step, _ in iterable_step:
            snippet_idx_ls = self.get_snippet_indice(
                i_step=i_step,
                timesteps=timesteps,
                seq_len=seq_len,
                snippet_len=snippet_len,
                dilation_start=start_dilation,
                dilation_end=1,
                stride=1,
            )

            t_current = timesteps[i_step]
            t_next = timesteps[i_step + 1] if i_step < len(timesteps) - 1 else None

            # Clear up new latent
            depth_latent_old = depth_latent_new.clone().detach()
            depth_latent_new *= 0.0
            count_depth_latent_new = torch.zeros(
                (*depth_latent_new.shape[:2], 1, 1, 1),
                device=depth_latent_new.device,
            ).int()  # [B, N, 1, 1, 1]

            # >>> Iterate through snippets >>>
            iterabel_snippet = snippet_idx_ls
            if verbose:
                iterabel_snippet = tqdm(
                    iterabel_snippet,
                    desc=" " * 4 + f"step {i_step+1} sliding windows",
                    leave=False,
                )
            for snippet_idx in iterabel_snippet:
                # Get input frames
                rgb_latent_input = rgb_latent[:, snippet_idx, :, :, :]
                # Use old latent
                depth_latent_input = depth_latent_old[:, snippet_idx, :, :, :]
                t_input = t_current.repeat(len(snippet_idx))

                # Denoising step
                noise_pred = self.single_step(
                    rgb_latent=rgb_latent_input,
                    depth_latent=depth_latent_input,
                    timestep=t_input,
                    encoder_hidden_states=batch_empty_text_embed,
                )
                _scheduler_output = self.scheduler.step(
                    noise_pred, t_current, depth_latent_input
                )
                depth_latent_pred = _scheduler_output.prev_sample

                # Cumulate new latents
                depth_latent_new[
                    :, snippet_idx, :, :, :
                ] += depth_latent_pred.clone().detach()
                count_depth_latent_new[:, snippet_idx, 0, 0, 0] += 1
            # <<< Iterate through snippets <<<

            # Average new latents
            assert torch.all(count_depth_latent_new > 0)
            depth_latent_new = depth_latent_new / count_depth_latent_new
            if t_next is not None:
                frame_timestep[:] = t_next

        return depth_latent_new

    def single_step(
        self,
        rgb_latent: torch.Tensor,
        depth_latent: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        #assert depth_latent.shape == rgb_latent.shape
        num_view = rgb_latent.shape[1]
        # assert 2 == num_view, f"only implemented for 2-view, got {rgb_latent.shape}"
        rgb_latent = einops.rearrange(rgb_latent, "b n c h w -> (b n) c h w")
        depth_latent = einops.rearrange(depth_latent, "b n c h w -> (b n) c h w")

        # Concat rgb and depth latents
        #unet_input = torch.cat([rgb_latent, depth_latent], dim=1)  # [N, 8, h, w]
        #**********************************
        unet_input = torch.cat([rgb_latent, depth_latent], dim=1)  # [N, 8, h, w]
        #print("unet_input shape:",unet_input.shape)
        # 使用零填充扩展到 8 通道
        # zeros = torch.zeros_like(rgb_latent)
        # unet_input = torch.cat([rgb_latent, zeros], dim=1)
        # sd15_name = 'models/stablediffusionapi-realistic-vision-v51'
        # self.unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

        device = torch.device('cuda')
        self.unet = self.unet.to(device=device, dtype=torch.float16)
        # unet_original_forward = self.unet.forward
        # def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        #     c_concat = kwargs['cross_attention_kwargs']['cross_attention_kwargs'].to(sample)
        #     c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        #     new_sample = torch.cat([sample, c_concat], dim=1)
        #     kwargs['cross_attention_kwargs'] = {}
        #     return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


        # self.unet.forward = hooked_unet_forward
        
        # self.cross_attention_kwargs={'concat_conds': rgb_latent}
        #conv_layer = nn.Conv2d(in_channels=16, out_channels=320, kernel_size=3, stride=1, padding=1)
        #***********************************
        #unet_input=unet_input.to(torch.float32)
        # predict the noise residual
        noise_pred = self.unet(
            unet_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            # timestep_cond=timestep_cond,
            # cross_attention_kwargs=self.cross_attention_kwargs,
            # added_cond_kwargs=added_cond_kwargs,
            #cross_attention_kwargs=self.cross_attention_kwargs,
            #added_cond_kwargs=added_cond_kwargs,
            num_view=num_view,
        ).sample  # [(B N) 4 h w]

        noise_pred = einops.rearrange(
            noise_pred, "(B N) C h w -> B N C h w", N=num_view
        )

        return noise_pred

    def encode_rgb(
        self, rgb_in: torch.Tensor, max_batch_size: int, verbose: bool = False
    ) -> torch.Tensor:
        self.vae = self.vae.to(self.device)
        #rgb_in = cv2.resize(rgb_in, (720, 640))
        if 5 == rgb_in.dim():
            B, N, _, H, W = rgb_in.shape
        else:
            B, _, H, W = rgb_in.shape
            N = 1

        rgb_in = einops.rearrange(rgb_in, "B N C H W -> (B N) C H W")

        # Process in batches
        latents = []
        iterable = range(0, B * N, max_batch_size)
        if verbose:
            iterable = tqdm(
                iterable,
                total=len(list(iterable)),
                leave=False,
                desc=" " * 4 + "Encoding",
            )
        for i in iterable:
            batch = rgb_in[i : i + max_batch_size]
            # encode
            h = self.vae.encoder(batch)
            moments = self.vae.quant_conv(h)
            mean, logvar = torch.chunk(moments, 2, dim=1)
            latents.append(mean)

        # Concatenate all batches
        rgb_latent = torch.cat(latents, dim=0)

        # scale latent
        rgb_latent = rgb_latent * self.rgb_latent_scale_factor

        h, w = rgb_latent.shape[-2:]  # latent shape
        rgb_latent = einops.rearrange(rgb_latent, "(B N) c h w -> B N c h w", B=B)

        return rgb_latent
    
    # def encode_fg_bg(
    #     self, fg_seq: torch.Tensor, bg_seq: torch.Tensor, max_batch_size: int, verbose: bool = False,
    #     prompt: str = "", a_prompt: str = "", n_prompt: str = ""
    # ) -> torch.Tensor:
    #     self.vae = self.vae.to(self.device)
        
    #     # 处理fg_seq和bg_seq的输入（进行维度等预处理示例，按需调整）
    #     if fg_seq is not None:
    #         fg_seq = [resize_and_center_crop(fg, fg_seq.shape[-2], fg_seq.shape[-1]) for fg in fg_seq]
    #         fg_seq = torch.stack(fg_seq).to(self.device).to(self.vae.dtype)
    #     if bg_seq is not None:
    #         bg_seq = [resize_and_center_crop(bg, bg_seq.shape[-2], bg_seq.shape[-1]) for bg in bg_seq]
    #         bg_seq = torch.stack(bg_seq).to(self.device).to(self.vae.dtype)
        
    #     # 对fg_seq和bg_seq的每一帧进行合并
    #     combined_fg_bg_seq = []
    #     for fg, bg in zip(fg_seq, bg_seq):
    #         #concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    #         concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor
    #         concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    #         conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    #         latents = t2i_pipe(
    #             prompt_embeds=conds,
    #             negative_prompt_embeds=unconds,
    #             width=image_width,
    #             height=image_height,
    #             num_inference_steps=steps,
    #             num_images_per_prompt=num_samples,
    #             generator=rng,
    #             output_type='latent',
    #             guidance_scale=cfg,
    #             cross_attention_kwargs={'concat_conds': concat_conds},
    #         ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

    #         pixels = self.vae.decode(latents).sample
    #         pixels = pytorch2numpy(pixels)
    #         pixels = [resize_without_crop(
    #             image=p,
    #             target_width=int(round(image_width * highres_scale / 64.0) * 64),
    #             target_height=int(round(image_height * highres_scale / 64.0) * 64))
    #         for p in pixels]

    #         pixels = numpy2pytorch(pixels).to(device=self.vae.device, dtype=self.vae.dtype)
    #         latents = self.vae.encode(pixels).latent_dist.mode() * self.vae.config.scaling_factor
    #         latents = latents.to(device=unet.device, dtype=unet.dtype)

    #         image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    #         fg = resize_and_center_crop(input_fg, image_width, image_height)
    #         bg = resize_and_center_crop(input_bg, image_width, image_height)
    #         concat_conds = numpy2pytorch([fg, bg]).to(device=self.vae.device, dtype=self.vae.dtype)
    #         concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor
    #         concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    #         latents = i2i_pipe(
    #             image=latents,
    #             strength=highres_denoise,
    #             prompt_embeds=conds,
    #             negative_prompt_embeds=unconds,
    #             width=image_width,
    #             height=image_height,
    #             num_inference_steps=int(round(steps / highres_denoise)),
    #             num_images_per_prompt=num_samples,
    #             generator=rng,
    #             output_type='latent',
    #             guidance_scale=cfg,
    #             cross_attention_kwargs={'concat_conds': concat_conds},
    #         ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
    #         concat_conds = concat_conds = numpy2pytorch([fg, bg]).to(device=self.vae.device, dtype=self.vae.dtype)
    #     combined_fg_bg_seq = torch.cat(combined_fg_bg_seq, dim=0)
    #     combined_fg_bg_seq = combined_fg_bg_seq.to(self.device).to(self.vae.dtype)

    #     # 利用vae进行编码等操作
    #     concat_conds = numpy2pytorch([combined_fg_bg_seq]).to(device=self.vae.device, dtype=self.vae.dtype)
    #     concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor
    #     concat_conds = torch.cat([c[None,...] for c in concat_conds], dim=1)

    #     # 获取conds和unconds（依赖外部定义的encode_prompt_pair函数，按给定提示生成）
    #     conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    #     # 此处可根据后续需要进一步处理concat_conds、conds和unconds之间的关系，比如结合用于生成latents等，示例中暂未深入展开那部分逻辑，假设已得到处理后的latents
    #     processed_latents_fg_bg = some_function_to_process_latents(concat_conds, conds, unconds)  # 需根据实际补充这个函数的定义

    #     # Process in batches
    #     latents = []
    #     iterable = range(0, processed_latents_fg_bg.shape[0], max_batch_size)
    #     if verbose:
    #         iterable = tqdm(
    #             iterable,
    #             total=len(list(iterable)),
    #             leave=False,
    #             desc=" " * 4 + "Encoding",
    #         )
    #     for i in iterable:
    #         batch = processed_latents_fg_bg[i : i + max_batch_size]
    #         # encode
    #         h = self.vae.encoder(batch)
    #         moments = self.vae.quant_conv(h)
    #         mean, logvar = torch.chunk(moments, 2, dim=1)
    #         latents.append(mean)

    #     # Concatenate all batches
    #     combined_latent = torch.cat(latents, dim=0)

    #     # 处理维度等（示例维度调整，按实际情况完善）
    #     combined_latent = einops.rearrange(combined_latent, "(B N) c h w -> B N c h w", B=fg_seq.shape[0])

    #     return combined_latent

    def decode_depth(
        self, depth_latent: torch.Tensor, max_batch_size: int, verbose: bool = False
    ) -> torch.Tensor:
        self.vae = self.vae.to(self.device)

        B, N, C, H, W = depth_latent.shape

        depth_latent = einops.rearrange(depth_latent, "b n c h w -> (b n) c h w")

        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor

        # Process in batches
        decoded_outputs = []
        iterable = range(0, B * N, max_batch_size)
        if verbose:
            iterable = tqdm(
                iterable,
                total=len(list(iterable)),
                leave=False,
                desc=" " * 4 + "Decoding",
            )
        for i in iterable:
            batch = depth_latent[i : i + max_batch_size]
            # decode
            z = self.vae.post_quant_conv(batch)
            stacked = self.vae.decoder(z)
            decoded_outputs.append(stacked)
        all_decoded = torch.cat(decoded_outputs, dim=0)

        # mean of output channels
        #depth_mean = all_decoded.mean(dim=1, keepdim=True)
       
        depth_mean = einops.rearrange(all_decoded, "(b n) c h w -> b n c h w", n=N)

        return depth_mean
    
    def decode_rgb(
        self, rgb_latent: torch.Tensor, max_batch_size: int, verbose: bool = False
    ) -> torch.Tensor:
        self.vae = self.vae.to(self.device)

        B, N, C, H, W = rgb_latent.shape

        # 展平批次和序列维度
        rgb_latent = einops.rearrange(rgb_latent, "b n c h w -> (b n) c h w")

        # 缩放 latent 特征
        rgb_latent = rgb_latent / self.rgb_latent_scale_factor

        # 分批解码
        decoded_outputs = []
        iterable = range(0, B * N, max_batch_size)
        if verbose:
            iterable = tqdm(
                iterable,
                total=len(list(iterable)),
                leave=False,
                desc=" " * 4 + "Decoding RGB",
            )
        for i in iterable:
            batch = rgb_latent[i : i + max_batch_size]
            # decode
            z = self.vae.post_quant_conv(batch)
            stacked = self.vae.decoder(z)
            decoded_outputs.append(stacked)

        # 合并所有批次
        all_decoded = torch.cat(decoded_outputs, dim=0)
        
        # 还原形状到 [B, N, C, H, W]
        decoded_rgb = einops.rearrange(all_decoded, "(b n) c h w -> b n c h w", n=N)
        print("decoded_rgb shape:",decoded_rgb[0].shape)

        return decoded_rgb

#@torch.inference_mode()
    def pytorch2numpy(imgs, quant=True):
        results = []
        for x in imgs:
            y = x.movedim(0, -1)

            if quant:
                y = y * 127.5 + 127.5
                y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
            else:
                y = y * 0.5 + 0.5
                y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

            results.append(y)
        return results


    #@torch.inference_mode()
    def numpy2pytorch(imgs):
        h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
        h = h.movedim(-1, 1)
        return h


    def resize_and_center_crop(image, target_width, target_height):
        pil_image = Image.fromarray(image)
        original_width, original_height = pil_image.size
        scale_factor = max(target_width / original_width, target_height / original_height)
        resized_width = int(round(original_width * scale_factor))
        resized_height = int(round(original_height * scale_factor))
        resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
        left = (resized_width - target_width) / 2
        top = (resized_height - target_height) / 2
        right = (resized_width + target_width) / 2
        bottom = (resized_height + target_height) / 2
        cropped_image = resized_image.crop((left, top, right, bottom))
        return np.array(cropped_image)


    def resize_without_crop(image, target_width, target_height):
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        return np.array(resized_image)
