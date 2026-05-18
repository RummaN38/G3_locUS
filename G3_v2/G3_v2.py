import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pyproj import Proj, Transformer

from rff.layers import GaussianEncoding

class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

class CustomLocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8]):
        super(CustomLocationEncoder, self).__init__()

        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

        proj_wgs84 = Proj('epsg:4326')
        proj_mercator = Proj('epsg:3857')
        self.transformer = Transformer.from_proj(proj_wgs84, proj_mercator, always_xy=True)

    def forward(self, input):
        device = input.device
        lat = input[:, 0].float().detach().cpu().numpy()
        lon = input[:, 1].float().detach().cpu().numpy()
        projected_lon_lat = self.transformer.transform(lon, lat)
        location = []
        for coord in zip(*projected_lon_lat):
            location.append([coord[1],coord[0]])
        location = torch.tensor(location, dtype=torch.float32, device=device)
        location = location / 20037508.3427892

        location_features = torch.zeros(location.shape[0], 512, device=device, dtype=torch.float32)

        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)

        return location_features


class G3(torch.nn.Module):
    def __init__(self, device):
        super(G3, self).__init__()
        self.device = device

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_model = clip_model.vision_model
        self.text_model = clip_model.text_model
        self.vision_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_projection = clip_model.visual_projection
        self.text_projection = clip_model.text_projection

        self.logit_scale1 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale2 = nn.Parameter(torch.tensor(3.99))
        self.logit_scale3 = nn.Parameter(torch.tensor(3.99))

        self.location_encoder = CustomLocationEncoder() # output batch_size, 3, 512
        #self.location_encoder = LocationEncoder(sigma=[2**0, 2**4, 2**8])
        self.vision_projection_else_1 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        self.text_projection_else = nn.Sequential(nn.Linear(768,768), nn.ReLU(), nn.Linear(768, 768))

        self.vision_projection_else_2 = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        self.location_projection_else = nn.Sequential(nn.Linear(512,512), nn.ReLU(), nn.Linear(512, 768))

        proj_dim = clip_model.config.projection_dim
        self.lstm_visual = nn.LSTM(
            input_size=proj_dim,
            hidden_size=proj_dim,
            batch_first=True,
            bidirectional=False,
            num_layers=1,
        )

        # 3×3×3 box filter, average per RGB channel
        _k = torch.full((3, 1, 3, 3, 3), 1.0 / 27.0)
        self.register_buffer("temporal_smooth_kernel", _k)

        # _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        # _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

        # m = torch.tensor(_CLIP_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1)
        # s = torch.tensor(_CLIP_STD, dtype=torch.float32).view(1, 1, 3, 1, 1)
        # self.register_buffer("_clip_norm_mean", m)
        # self.register_buffer("_clip_norm_std", s)

        # freeze CLIP
        self.vision_model.requires_grad_(False)
        self.vision_projection.requires_grad_(False)
        self.text_model.requires_grad_(False)
        self.text_projection.requires_grad_(False)

    def _encode_video(self, video, video_mask=None):
        # apply CLIP mean/std -- already done in the RawVideoExtractor
        # x = video.float()
        # mean = self._clip_norm_mean.to(dtype=x.dtype, device=x.device)
        # std = self._clip_norm_std.to(dtype=x.dtype, device=x.device)
        # video = (x - mean) / std

        if video.dim() == 7:
            video = video.squeeze(1).squeeze(2)
        elif video.dim() == 6 and video.shape[2] == 1:
            video = video.squeeze(2)
        
        if video_mask.dim() == 3:
            video_mask = video_mask.squeeze(1)

        if video.dim() == 4:
            video = video.unsqueeze(1)
        if video.dim() != 5:
            raise ValueError("video must be [B, T, 3, H, W] or [B, 3, H, W]")

        b, t, c, h, w = video.shape
        if video_mask is None:
            video_mask = torch.ones(b, t, device=video.device, dtype=torch.long)
        else:
            video_mask = video_mask.to(device=video.device)
            if video_mask.shape != (b, t):
                raise ValueError(f"video_mask must be [B, T] = [{b}, {t}]")

        # 3d conv with spatial and temporal smoothing
        orig_dtype = video.dtype
        x3d = video.float().permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, H, W]
        k_smooth = self.temporal_smooth_kernel.to(device=x3d.device, dtype=x3d.dtype)
        x3d = F.pad(x3d, (1, 1, 1, 1, 1, 1), mode="replicate")
        x3d = F.conv3d(x3d, k_smooth, bias=None, stride=1, padding=(0, 0, 0), groups=3)
        video = x3d.permute(0, 2, 1, 3, 4).to(dtype=orig_dtype)

        flat = video.reshape(b * t, c, h, w) # [B * T, 3, H, W]
        vision_out = self.vision_model(pixel_values=flat)
        pooler_flat = vision_out.pooler_output # [B * T, 768]
        vision_seq = pooler_flat.reshape(b, t, -1) # [B, T, 768]
        frame_embeds = self.vision_projection(pooler_flat).reshape(b, t, -1) # [B, T, 768]

        lengths = video_mask.sum(dim=1).to(torch.long).cpu()

        packed = pack_padded_sequence(frame_embeds, lengths, batch_first=True, enforce_sorted=False) # [sum_i lengths[i], proj_dim]
        lstm_out, _ = self.lstm_visual(packed) # [sum_i lengths[i], proj_dim]
        if self.training:
            self.lstm_visual.flatten_parameters()
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=t) # [B, T, proj_dim]
        mask_f = video_mask.to(dtype=lstm_out.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        video_embeds = (lstm_out * mask_f).sum(dim=1) / denom # [B, proj_dim] - average of valid frames of LSTM embeddings

        mask_v = video_mask.to(dtype=vision_seq.dtype).unsqueeze(-1)
        vision_pooled_mean = (vision_seq * mask_v).sum(dim=1) / mask_v.sum(dim=1).clamp(min=1.0) # [B, 768] - average of valid frames of CLIP embeddings

        return video_embeds, vision_pooled_mean

    def forward(
        self,
        video, # [B, T, 3, H, W] - B - batch size, T - number of frames, 3 - channels, H - height, W - width
        texts,
        longitude,
        latitude,
        return_loss=True,
        video_mask=None
    ):
        video_embeds, vision_pooled_mean = self._encode_video(video, video_mask)
        text_output = self.text_model(**texts)[1]
        text_embeds = self.text_projection(text_output) # batch_size, 512
        this_batch_locations = torch.stack((latitude, longitude), dim=1)
        location_embeds = self.location_encoder(this_batch_locations)

        # phase _1 - text with video
        video_embeds_1 = self.vision_projection_else_1(video_embeds)
        text_embeds_1 = self.text_projection_else(text_embeds.reshape(text_embeds.shape[0], -1))
        
        # normalized features
        video_embeds_1 = video_embeds_1 / video_embeds_1.norm(p=2, dim=-1, keepdim=True)
        text_embeds_1 = text_embeds_1 / text_embeds_1.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale1.exp()
        logits_per_texts_with_videos = torch.matmul(text_embeds_1, video_embeds_1.t()) * logit_scale
        logits_per_videos_with_texts = logits_per_texts_with_videos.t()
        if return_loss: loss1 = self.clip_loss(logits_per_texts_with_videos)

        loss_phase_1 = None
        if return_loss:
            loss_phase_1 = loss1

        # phase _2 - location with video
        video_embeds_2 = self.vision_projection_else_2(video_embeds)
        location_embeds_2 = self.location_projection_else(location_embeds.reshape(location_embeds.shape[0], -1))

        # normalized features
        video_embeds_2 = video_embeds_2 / video_embeds_2.norm(p=2, dim=-1, keepdim=True)
        location_embeds_2 = location_embeds_2 / location_embeds_2.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale2.exp()
        logits_per_locations_with_videos = torch.matmul(location_embeds_2, video_embeds_2.t()) * logit_scale
        logits_per_videos_with_locations = logits_per_locations_with_videos.t()
        loss_phase_2 = None
        if return_loss: loss_phase_2 = self.clip_loss(logits_per_locations_with_videos)

        loss = loss_phase_1 + loss_phase_2

        return {
            'logits_per_texts_with_videos': logits_per_texts_with_videos,
            'logits_per_videos_with_texts': logits_per_videos_with_texts,
            'logits_per_locations_with_videos': logits_per_locations_with_videos,
            'logits_per_videos_with_locations': logits_per_videos_with_locations,
            'logits_per_locations_with_texts': None,
            'logits_per_texts_with_locations': None,
            'loss': loss,
            'vision_output': vision_pooled_mean,
            'text_output': text_output,
            'video_embeds': video_embeds,
            'text_embeds': text_embeds
        }


    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        loss_a = self.contrastive_loss(similarity)
        loss_b = self.contrastive_loss(similarity.t())
        return (loss_a + loss_b) / 2.0
