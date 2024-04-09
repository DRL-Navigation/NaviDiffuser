import torch, tqdm, os, datetime, yaml
from nn import GaussianDiffusion
from nn import TemporalUnet
from data import NaviDataset, NaviCollateFN
from torch.utils.data import DataLoader as DL
from utils import Trainer

def read_yaml(file: str) -> dict:
    with open(file, 'r', encoding="utf-8") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data

if __name__ == "__main__":
    cfg = read_yaml('./cfg/train.yaml')
    model = TemporalUnet(**cfg['TemporalUnet'])
    Diffusion_model = GaussianDiffusion(model=model, **cfg['GaussianDiffusion'])
    dataset = NaviDataset(cfg['Dataset']['path'])
    collate = NaviCollateFN(**cfg['Dataset']['CollateFN'])
    dataloader = DL(dataset, batch_size=cfg['Dataset']['batch'], num_workers=1, shuffle=True, pin_memory=True, collate_fn=collate.collate_fn)
    trainer = Trainer(diffusion_model=Diffusion_model, dataset=dataset, dataloader=dataloader, **cfg['Trainer'])

    trainer.train(cfg['n_train_steps'])

