from datetime import datetime
import json
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import yaml

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
import torchvision.models as models

sys.path.append('../')
from dew_dataset import DewDataset

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

#CONST
    with open('../config.yml') as yml:
        config = yaml.load(yml, Loader=yaml.SafeLoader)
        try:
            config = config[os.uname()[1]]
        except Exception as e:
            config = config['aws']

    obj = 'full'
    data_size = 'all'
    if rank == 0:
        now = datetime.now()
        PATH = os.path.join(config['out_dir_base'],
                        os.path.splitext(os.path.basename(__file__))[0] +
                        '_' + obj +
                        '_' + data_size +
                        '_' + now.strftime("%Y%m%d-%H%M%S"))

        if not os.path.exists(PATH):
            os.mkdir(PATH)
        LOGFILE = os.path.join(PATH, 'training.log')

        header = []
        header.append('-'*50)
        header.append('PyTorch Version: %s' % torch.__version__)
        header.append('DEVICE COUNT: %s' % torch.cuda.device_count())
        header.append(json.dumps(config, indent=4))
        header.append('Output Path: %s' % PATH)
        header.append('-'*50)

        with open(LOGFILE, 'w') as f:
            for entry in header:
                print(entry)
                f.write('%s\n' % entry)
                f.flush()

    # MODEL
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.set_device(rank)

    model = models.resnet50(pretrained=True, progress=True)
    model.fc = nn.Linear(2048, 1, bias=True)
    ddp_model = DDP(model.to(rank), device_ids=[rank])

    torch.backends.cudnn.deterministic = True
    cost_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=config['learning_rate']['reg'])

    # Dataset
    train_transform = transforms.Compose([transforms.Resize(config['resize']),
                                    transforms.RandomCrop(config['crop']),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    eval_transform = transforms.Compose([transforms.Resize(config['resize']),
                                    transforms.CenterCrop(config['crop']),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    train_csv = os.path.join(config['base_dir'], config['csv'][obj][data_size])
    valid_csv = os.path.join(config['base_dir'], config['csv'][obj]['valid'])
    train_dataset = DewDataset(csv_path=train_csv, img_dir=config['image_dir'], transform=train_transform)
    valid_dataset = DewDataset(csv_path=valid_csv, img_dir=config['image_dir'], transform=eval_transform)
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=int(config['batch_size']/world_size), sampler=sampler, num_workers=4*world_size, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=int(config['batch_size']/world_size), shuffle=False, num_workers=4*world_size, pin_memory=True)

    if rank == 0:
        writer = SummaryWriter(log_dir='../runs/{obj}-Regression-{now}'.format(obj=obj, lr=config['learning_rate']['reg'], now=now.strftime("%Y%m%d-%H%M%S")))
        print(ddp_model)

    scaler = torch.cuda.amp.GradScaler()

    best_mae = float('inf')
    checkpoint = None
    bestpoint = None
    counter = 0
    for epoch in range(config['epochs']):
        if epoch * len(train_loader) >= 1000000:
            break

        if epoch != 0:
            time_elapsed = finish_time - start_time
            print(time_elapsed)
            with open(LOGFILE, 'a') as f:
                f.write('\n%s\n' % ('Time elapsed ' + str(time_elapsed)))

        start_time = time.time()
        sampler.set_epoch(epoch)

        if rank == 0:
            print('Epoch ', epoch)
            with open(LOGFILE, 'a') as f:
                f.write('\n%s\n' % ('Epoch ' + str(epoch)))
            bar = tqdm(total=len(train_loader))
            mae, mse, num_examples = 0, 0, 0

        for idx, (features, targets, _) in enumerate(train_loader):
            ddp_model.train()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                estimate_label = ddp_model(features)
                targets = targets.to(rank)
                cost = cost_fn(estimate_label.flatten(), targets.float())

            scaler.scale(cost).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                bar.update(1)
                num_examples += targets.size(0)
                mae += torch.sum(torch.abs(estimate_label.flatten() - targets.float()))
                mse += torch.sum((estimate_label.flatten() - targets.float())**2)

                iteration = epoch * len(train_loader) + idx
                if iteration % 100 == 0:
                    mae = mae.float() / num_examples
                    mse = mse.float() / num_examples

                    with open(LOGFILE, 'a') as f:
                        f.write('\n%s\n' % ('Iteration '+ str(iteration)))
                        f.write('TRAIN %s\n' % 'MAE/RMSE: %.2f/%.2f' % (mae, torch.sqrt(mse)))

                    writer.add_scalar('Cost/train', cost, counter)
                    writer.add_scalar('MAE/train', mae, counter)
                    writer.add_scalar('MSE/train', torch.sqrt(mse), counter)

                    # test
                    ddp_model.eval()
                    with torch.set_grad_enabled(False):  # save memory during inference
                        mae, mse, num_examples = 0, 0, 0
                        for i, (features, targets, _) in enumerate(valid_loader):
                            estimate_label = ddp_model(features)
                            targets = targets.to(rank)

                            num_examples += targets.size(0)
                            mae += torch.sum(torch.abs(estimate_label.flatten() - targets.float()))
                            mse += torch.sum((estimate_label.flatten() - targets.float())**2)
                        mae = mae.float() / num_examples
                        mse = mse.float() / num_examples

                    print('Iteration '+ str(iteration))
                    eval_text = 'VALID %s' % 'MAE/RMSE: %.2f/%.2f' % (mae, torch.sqrt(mse))
                    print(eval_text)

                    with open(LOGFILE, 'a') as f:
                        f.write('%s\n' % eval_text)

                    # tensorboard
                    writer.add_scalar('MAE/valid', mae, counter)
                    writer.add_scalar('MSE/valid', torch.sqrt(mse), counter)

                    if mae < best_mae:
                        break_str = 'Break the best score Iteration ' + str(iteration)
                        print(break_str)
                        with open(LOGFILE, 'a') as f:
                            f.write('%s\n' % (break_str))

                        best_mae = mae
                        if bestpoint is not None:
                            os.remove(bestpoint)
                        bestpoint = os.path.join(PATH, 'best_{}.pt'.format(iteration))
                        torch.save({
                                'iteration' : iteration,
                                'model_state_dict' : ddp_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scaler': scaler.state_dict(),
                                'loss': cost
                        }, bestpoint)

                    if checkpoint is not None:
                        os.remove(checkpoint)
                    checkpoint = os.path.join(PATH, 'checkpoint_{}.pt'.format(iteration))
                    torch.save({
                            'iteration' : iteration,
                            'model_state_dict' : ddp_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler': scaler.state_dict(),
                            'loss': cost
                    }, checkpoint)

                    mae, mse, num_examples = 0, 0, 0
                    counter += 1
                    finish_time = time.time()

    # test
    test_csv = os.path.join(config['base_dir'], config['csv'][obj]['test'])
    test_dataset = DewDataset(csv_path=test_csv, img_dir=config['image_dir'], transform=eval_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=int(config['batch_size']/world_size), shuffle=False, num_workers=4*world_size, pin_memory=True)

    ddp_model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        mae, mse, num_examples = 0, 0, 0
        for i, (features, targets, _) in enumerate(test_loader):
            estimate_label = ddp_model(features)
            targets = targets.to(rank)

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(estimate_label.flatten() - targets.float()))
            mse += torch.sum((estimate_label.flatten() - targets.float())**2)
        mae = mae.float() / num_examples
        mse = mse.float() / num_examples

    print('-'*50)
    print('TEST')
    eval_text = 'TEST %s' % 'MAE/RMSE: %.2f/%.2f' % (mae, torch.sqrt(mse))
    print(eval_text)

    with open(LOGFILE, 'a') as f:
        f.write('TEST\n')
        f.write('%s\n' % eval_text)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)
