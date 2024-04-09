import torch, tqdm, os, yaml
import torch.utils.tensorboard as tensorboard
from torch.utils.data import DataLoader as DL
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from data import NaviDataset, NaviCollateFN
from nn import NaviDiffusion, ClassifyFreeDDIM

def read_yaml(file: str) -> dict:
    with open(file, 'r', encoding="utf-8") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data

class Train:
    def __init__(self, model, dataloader, optimizer, scaler, scheduler, rank=0, device="cuda", world_size=1, epoch=10000, grad_norm_clip=None, fine_tune=None, name="model"):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.epoch = epoch
        self.grad_norm_clip = grad_norm_clip
        if fine_tune != None:
            self.model.load(fine_tune)
        self.tb_writer = None
        self.name = name
        self.rank = rank
        self.device = device
        self.world_size = world_size

    def _to_device(self, data):
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = self._to_device(data[i])
        elif isinstance(data, dict):
            for key in data.keys():
                data[key] = self._to_device(data[key])
        elif isinstance(data, torch.Tensor):
            data = data.to(device=self.device, non_blocking=True)
        else:
            data = data
        return data

    def _run(self, data, epoch)->None:
        self.model.train()
        self.optimizer.zero_grad()
        with autocast():
            loss = self.model.learn(*data)
        self.scaler.scale(loss).backward()
        if self.grad_norm_clip != None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.tb_writer != None:
            self.tb_writer.add_scalar('Loss', loss.item(), epoch)

    def Run(self)->None:
        if self.rank == 0:
            print('--Training Process Running:', flush=True)
            if not os.path.exists('./log'): os.mkdir('./log')
            if not os.path.exists('./log/{}'.format(self.name)): os.mkdir('./log/{}'.format(self.name))
            self.tb_writer = tensorboard.SummaryWriter(log_dir='./log/{}/tfboard'.format(self.name))

        data_iter = iter(self.dataloader)
        epoch_iter = range(self.epoch) if self.rank != 0 else tqdm.tqdm(range(self.epoch), '', ncols=120, unit='Epoch')
        for epoch in epoch_iter:
            data = next(data_iter, None)
            if data is None:
                data_iter = iter(self.dataloader)
                data = next(data_iter)
            data = self._to_device(data)
            self._run(data, epoch)
            if (self.rank == 0) and ((epoch+1) % 5000 == 0):
                self.model.save('./log/{}/'.format(self.name)+str(int((epoch+1)/1000))+'K.pt')

        if self.rank == 0:
            self.model.save('./log/{}/last_model.pt'.format(self.name))
            print('--Finish: Models are saved in ./log/{} .'.format(self.name), flush=True)
            

if __name__ == "__main__":
    torch.backends.cudnn.enable=True
    torch.backends.cudnn.benchmark=True

    rank, world_size, device = 0, 1, 'cuda'
    cfg = read_yaml('./cfg/train.yaml')

    dataset = NaviDataset(cfg['Dataset']['path'])
    collate = NaviCollateFN(**cfg['Dataset']['CollateFN'])
    dataloader = DL(dataset=dataset, batch_size=cfg['Dataset']['batch'], num_workers=os.cpu_count()//world_size, shuffle=True, collate_fn=collate.collate_fn, drop_last=True, pin_memory=True)
    model = NaviDiffusion(**cfg['Network']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['Optim']['lr'], weight_decay=cfg['Optim']['weight_decay'])
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/cfg['Optim']['warm_up'], 1))
    model = ClassifyFreeDDIM(model, **cfg['Diffusion'])

    train = Train(model, dataloader, optimizer, scaler, scheduler, rank, device, world_size, **cfg['Train'])
    train.Run()
