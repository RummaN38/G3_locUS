import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .rff.layers import GaussianEncoding

from pyproj import Proj, Transformer

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

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
        lat = input[:, 0].float().detach().cpu().numpy()
        lon = input[:, 1].float().detach().cpu().numpy()
        projected_lon_lat = self.transformer.transform(lon, lat)
        location = []
        for coord in zip(*projected_lon_lat):
            location.append([coord[1],coord[0]])
        location = torch.Tensor(location).to('cuda')
        location = location / 20037508.3427892

        location_features = torch.zeros(location.shape[0], 512).to('cuda')

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

        m = torch.tensor(_CLIP_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1)
        s = torch.tensor(_CLIP_STD, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("_clip_norm_mean", m)
        self.register_buffer("_clip_norm_std", s)

        # freeze CLIP
        self.vision_model.requires_grad_(False)
        self.vision_projection.requires_grad_(False)
        self.text_model.requires_grad_(False)
        self.text_projection.requires_grad_(False)

    def _vision_pooler(self, pixel_values_bt):
        """pixel_values_bt: [N, 3, H, W] in the same value range as HF CLIP vision_model expects."""
        out = self.vision_model(pixel_values=pixel_values_bt)
        return out.pooler_output

    def _maybe_apply_clip_normalize(self, video, apply_clip_normalization):
        """If True, treat video as linear RGB in [0, 1] (e.g. after ToTensor) and apply CLIP mean/std."""
        if not apply_clip_normalization:
            return video
        x = video.float()
        if x.shape[-1] <= 4 or x.shape[-2] <= 4:
            raise ValueError("Expected video shape [B, T, 3, H, W] with H, W > 4 for normalization.")
        mean = self._clip_norm_mean.to(dtype=x.dtype, device=x.device)
        std = self._clip_norm_std.to(dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def _encode_video_to_clip_space(self, video, video_mask=None, apply_clip_normalization=False):
        """
        Encode a clip with frozen CLIP ViT (per frame) + trainable LSTM + masked mean over LSTM outputs.

        No residual to original frame embeddings: the clip vector is only from temporal LSTM states.

        video: [B, T, 3, H, W] — if apply_clip_normalization=True, values in [0, 1] per channel
               (resize/center-crop to model size outside this module, e.g. via CLIPImageProcessor);
               if False, pass tensors exactly as HF CLIP vision_model expects (processor output).
        video_mask: optional [B, T] with 1 for valid frames, 0 for padding.
        Returns:
            video_embeds: [B, projection_dim] — mean of per-timestep LSTM outputs on valid frames
            vision_pooled_mean: [B, vision_hidden] mean of pooler outputs (for debugging / returns)
        """
        video = self._maybe_apply_clip_normalize(video, apply_clip_normalization)
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

        flat = video.reshape(b * t, c, h, w)
        pooler_flat = self._vision_pooler(flat)
        vision_seq = pooler_flat.reshape(b, t, -1)
        frame_embeds = self.vision_projection(pooler_flat).reshape(b, t, -1)

        lengths = video_mask.sum(dim=1).to(torch.long).cpu()
        if (lengths == 0).any():
            raise ValueError("video_mask must have at least one valid frame per batch row.")

        packed = pack_padded_sequence(frame_embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm_visual(packed)
        if self.training:
            self.lstm_visual.flatten_parameters()
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=t)
        mask_f = video_mask.to(dtype=lstm_out.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        video_embeds = (lstm_out * mask_f).sum(dim=1) / denom

        mask_v = video_mask.to(dtype=vision_seq.dtype).unsqueeze(-1)
        vision_pooled_mean = (vision_seq * mask_v).sum(dim=1) / mask_v.sum(dim=1).clamp(min=1.0)

        return video_embeds, vision_pooled_mean

    def forward(
        self,
        video,
        texts,
        longitude,
        latitude,
        return_loss=True,
        video_mask=None,
        apply_clip_normalization=False,
    ):
        """
        video: [B, 3, H, W] (single frame) or clip [B, T, 3, H, W].
        Contrastive phases use one embedding per clip: masked mean of LSTM outputs (no residual to CLIP frame vectors).
        video_mask: [B, T] when sequences are padded.
        apply_clip_normalization: if True, apply CLIP mean/std (for [0,1] RGB after resize/crop).
        """
        video_embeds, vision_pooled_mean = self._encode_video_to_clip_space(
            video, video_mask=video_mask, apply_clip_normalization=apply_clip_normalization
        )
        text_output = self.text_model(**texts)[1]
        text_embeds = self.text_projection(text_output) # batch_size, 512
        this_batch_locations = torch.stack((latitude, longitude), dim=1)
        location_embeds = self.location_encoder(this_batch_locations)

        # phase _1 — text ↔ video (clip-level)
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

        # phase _2 — location ↔ video (clip-level)
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
