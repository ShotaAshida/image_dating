import argparse
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

#from image_dating.model.dew_dataset import DewCoralDataset
#from image_dating.model.coral.coral_model import Coral
sys.path.append('..')
from dew_dataset import DewCoralPersonDataset
from coral_model import Coral

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    #CONST
    with open('../config.yml') as yml:
        config = yaml.load(yml, Loader=yaml.SafeLoader)
        config = config[os.uname()[1]]

    obj = 'person'
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

    model = Coral()
    pre_trained = models.resnet50(pretrained=True, progress=True).state_dict()
    del pre_trained['fc.weight']
    del pre_trained['fc.bias']
    model.load_state_dict(pre_trained, strict=False)
    ddp_model = DDP(model.to(rank), device_ids=[rank])

    torch.backends.cudnn.deterministic = True
#    optimizer = optim.SGD([{'params': other_params}, {'params': fc_param, 'lr': config['fc_learning_rate']}], lr=config['learning_rate'], momentum=config['momentum'])
#    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
    optimizer = optim.Adam(ddp_model.parameters(), lr=config['learning_rate']['coral'])

    # Dataset
    train_transform = transforms.Compose([transforms.Resize((config['resize'], config['resize'])),
                                    transforms.RandomCrop(config['crop']),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    eval_transform = transforms.Compose([transforms.Resize((config['resize'], config['resize'])),
                                    transforms.CenterCrop(config['crop']),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    train_csv = os.path.join(config['base_dir'], config['csv'][obj][data_size])
    valid_csv = os.path.join(config['base_dir'], config['csv'][obj]['valid'])
    train_dataset = DewCoralPersonDataset(csv_path=train_csv, img_dir=config['image_dir'], transform=train_transform, reverse=True)
    valid_dataset = DewCoralPersonDataset(csv_path=valid_csv, img_dir=config['image_dir'], transform=eval_transform, reverse=True)
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=int(config['batch_size']/world_size), sampler=sampler, num_workers=4*world_size, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=int(config['batch_size']/world_size), shuffle=False, num_workers=4*world_size, pin_memory=True)

    if rank == 0:
        writer = SummaryWriter(log_dir='../runs/Person-ReverseCoral-Adam-valid-lr{lr}'.format(lr=config['learning_rate']['coral']))
        print(ddp_model)

    # Train
    best_mae = float('inf')
    checkpoint = None
    bestpoint = None
    counter = 0
    for epoch in range(config['epochs']):
        if epoch * len(train_loader) >= 1000000:
            break

        start_time = time.time()
        sampler.set_epoch(epoch)

        if rank == 0:
            print('Epoch ', epoch)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % ('Epoch ' + str(epoch)))
            bar = tqdm(total=len(train_loader))
            mae, mse, num_examples = 0, 0, 0

        for idx, (features, targets, levels, _) in enumerate(train_loader):
            ddp_model.train()
            logits, probas = ddp_model(features)
            cost = cost_fn(logits, levels.to(rank))
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if rank == 0:
                bar.update(1)
                estimated_levels = probas > 0.5
                estimated_labels = 69 - torch.sum(estimated_levels, dim=1)
                targets = targets.to(rank)
                num_examples += targets.size(0)
                mae += torch.sum(torch.abs(estimated_labels - targets))
                mse += torch.sum((estimated_labels - targets)**2)

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
                        for i, (features, targets, levels, _) in enumerate(valid_loader):
                            logits, probas = ddp_model(features)
                            estimated_levels = probas > 0.5
                            estimated_labels = 69 - torch.sum(estimated_levels, dim=1)
                            targets = targets.to(rank)

                            num_examples += targets.size(0)
                            mae += torch.sum(torch.abs(estimated_labels - targets))
                            mse += torch.sum((estimated_labels - targets)**2)
                        mae = mae.float() / num_examples
                        mse = mse.float() / num_examples

                    print('Iteration '+ str(iteration))
                    eval_text = 'VALID %s' % 'MAE/RMSE: %.2f/%.2f' % (mae, torch.sqrt(mse))
                    print(eval_text)
                    time_text = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
                    print(time_text)

                    with open(LOGFILE, 'a') as f:
                        f.write('%s\n' % eval_text)
                        f.write('%s\n' % time_text)

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
                                'loss': cost
                        }, bestpoint)

                    if checkpoint is not None:
                        os.remove(checkpoint)
                    checkpoint = os.path.join(PATH, 'checkpoint_{}.pt'.format(iteration))
                    torch.save({
                            'iteration' : iteration,
                            'model_state_dict' : ddp_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': cost
                    }, checkpoint)

                    mae, mse, num_examples = 0, 0, 0
                    counter += 1

    # test
    test_csv = os.path.join(config['base_dir'], config['csv'][obj]['test'])
    test_dataset = DewCoralObjectDataset(csv_path=test_csv, img_dir=config['image_dir'], transform=eval_transform, reverse=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=int(config['batch_size']/world_size), shuffle=False, num_workers=4*world_size, pin_memory=True)

    ddp_model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        mae, mse, num_examples = 0, 0, 0
        for i, (features, targets, levels, _) in enumerate(test_loader):
            logits, probas = ddp_model(features)
            estimated_levels = probas > 0.5
            estimated_labels = torch.sum(estimated_levels, dim=1)
            targets = targets.to(rank)

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(estimated_labels - targets))
            mse += torch.sum((estimated_labels - targets)**2)
        mae = mae.float() / num_examples
        mse = mse.float() / num_examples

    print('-'*50)
    print('TEST')
    eval_text = 'TEST %s' % 'MAE/RMSE: %.2f/%.2f' % (mae, torch.sqrt(mse))
    print(eval_text)
    time_text = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(time_text)

    with open(LOGFILE, 'a') as f:
        f.write('TEST\n')
        f.write('%s\n' % eval_text)
        f.write('%s\n' % time_text)

def cost_fn(logits, levels):
    val = (-torch.sum((F.logsigmoid(logits)*levels + (F.logsigmoid(logits) - logits)*(1-levels)), dim=1))
    return torch.mean(val)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)

