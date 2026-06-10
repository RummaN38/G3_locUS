import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings

from config import load_config
from G3_v2 import G3
from sekai_dataset import SEKAI_Real_Walking_Dataset


warnings.filterwarnings('ignore')


def train_1epoch(dataloader, eval_dataloader, earlystopper, model, vision_processor, text_processor, optimizer, scheduler, device, accelerator=None):
    model.train()
    t = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    for i, (texts, images, video_mask, latitude, longitude) in enumerate(t):
        texts = text_processor(text=texts, padding='max_length', truncation=True, return_tensors='pt', max_length=77)
        images = images.to(device)
        texts = texts.to(device)
        video_mask = video_mask.to(device)
        longitude = longitude.to(device).float()
        latitude = latitude.to(device).float()
        optimizer.zero_grad()

        output = model(images, texts, longitude, latitude, video_mask=video_mask, return_loss=True)
        loss = output['loss']

        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        if i % 1 == 0:
            t.set_description('step {}, loss {}, lr {}'.format(i, loss.item(), scheduler.get_last_lr()[0]))
    scheduler.step()


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # fine-tune
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = G3(device).to(device)
    location_encoder_dict = torch.load('location_encoder.pth') # from geoclip
    model.location_encoder.load_state_dict(location_encoder_dict)

    dataset = SEKAI_Real_Walking_Dataset(
        csv_path=data_cfg["csv_path"],
        features_path=data_cfg["features_path"],
        feature_framerate=data_cfg["feature_framerate"],
        max_frames=data_cfg["max_frames"],
        image_resolution=data_cfg["image_resolution"],
        save_frames=data_cfg["save_frames"],
        yamnet_hop=data_cfg["yamnet_hop"],
        refresh_metadata=data_cfg["refresh_metadata"],
    )
    loader_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )
    if train_cfg["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = train_cfg["prefetch_factor"]
    dataloader = DataLoader(dataset, **loader_kwargs)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())

    optimizer = torch.optim.AdamW([param for name,param in model.named_parameters() if param.requires_grad], lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg["scheduler_step_size"], gamma=train_cfg["scheduler_gamma"])

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    device = accelerator.device

    eval_dataloader = None
    earlystopper = None
    for epoch in tqdm(range(train_cfg["epochs"])):
        train_1epoch(dataloader, eval_dataloader, earlystopper, model, model.vision_processor, model.text_processor, optimizer, scheduler, device, accelerator)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model, 'checkpoints/g3_{}.pth'.format(epoch))
        torch.save(unwrapped_model.state_dict(), 'checkpoints/g3_{}_state_dict.pth'.format(epoch))

if __name__ == '__main__':
    main()
