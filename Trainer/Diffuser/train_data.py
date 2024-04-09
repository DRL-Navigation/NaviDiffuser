import torch, tqdm, os, datetime, yaml
from nn import ValueDiffusion
from nn import ValueFunction
from data import NaviDataset, NaviValueCollateFN
from torch.utils.data import DataLoader as DL
from utils import Trainer

def read_yaml(file: str) -> dict:
    with open(file, 'r', encoding="utf-8") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data

if __name__ == "__main__":
    cfg = read_yaml('./cfg/train.yaml')
    model = ValueFunction(**cfg['ValueFunction'])
    Diffusion_model = ValueDiffusion(model=model, **cfg['ValueDiffusion'])
    dataset = NaviDataset(cfg['Dataset']['path'])
    collate = NaviValueCollateFN(**cfg['Dataset']['ValueCollateFN'])
    dataloader = DL(dataset, batch_size=cfg['Dataset']['batch'], num_workers=1, shuffle=True, pin_memory=True, collate_fn=collate.collate_fn)
    trainer = Trainer(diffusion_model=Diffusion_model, dataset=dataset, dataloader=dataloader, **cfg['Trainer'])

    trainer.train(cfg['n_train_steps'])

