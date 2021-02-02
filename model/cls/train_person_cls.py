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

sys.path.append('../')
from dew_dataset import DewPersonDataset
#from image_dating.model.dew_dataset import DewDataset
#from image_dating.bot import slack_bot as sb

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    #CONST
    with open('../config.yml') as yml:
        config = yaml.load(yml, Loader=yaml.SafeLoader)
        config = config[os.uname()[1]]

    data_size = 'all'
    if rank == 0:
        now = datetime.now()
        PATH = os.path.join(config['out_dir_base'],
                        os.path.splitext(os.path.basename(__file__))[0] +
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
    if config['feature_extract']:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(2048, config['class_num'], bias=True)
    ddp_model = DDP(model.to(rank), device_ids=[rank])

    params_to_update = ddp_model.parameters()
    print("Params to learn:")
    if config['feature_extract']:
        params_to_update = []
        for name,param in ddp_model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        print('finetuning')

    torch.backends.cudnn.deterministic = True
    cost_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_to_update, lr=config['learning_rate'])

    # Dataset
    train_transform = transforms.Compose([transforms.Resize(config['crop']),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    test_transform = transforms.Compose([transforms.Resize(config['crop']),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    train_csv = os.path.join(config['base_dir'], config['csv']['person'][data_size])
    test_csv = os.path.join(config['base_dir'], config['csv']['person']['test'])
    train_dataset = DewPersonDataset(csv_path=train_csv, img_dir=config['image_dir'], transform=train_transform)
    test_dataset = DewPersonDataset(csv_path=test_csv, img_dir=config['image_dir'], transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=int(config['batch_size']/world_size), shuffle=False)
    sampler = DistributedSampler(train_dataset)

    if rank == 0:
        writer = SummaryWriter(comment="-person")
        print(ddp_model)

    # Train
    best_mae = float('inf')
    checkpoint = None
    counter = 0
    for epoch in range(config['epochs']):
        start_time = time.time()

        # train
        sampler.set_epoch(epoch)
        dataloader = DataLoader(dataset=train_dataset, batch_size=int(config['batch_size']/world_size), sampler=sampler)

        if rank == 0:
            print('Epoch', epoch)
            with open(LOGFILE, 'a') as f:
                f.write('\n%s\n' % ('Epoch ' + str(epoch)))
            bar = tqdm(total=len(dataloader))
            mae, mse, num_examples = 0, 0, 0

        for idx, (features, targets, _) in enumerate(dataloader):
            ddp_model.train()
            outputs = ddp_model(features)
            cost = cost_fn(outputs, targets.to(rank))
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if rank == 0:
                bar.update(1)
                estimated_labels = torch.max(outputs, 1)[1]
                targets = targets.to(rank)
                num_examples += targets.size(0)
                mae += torch.sum(torch.abs(estimated_labels - targets))
                mse += torch.sum((estimated_labels - targets)**2)

                iteration = epoch * len(dataloader) + idx
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
                        for i, (features, targets, _) in enumerate(test_loader):
                            outputs = ddp_model(features)
                            estimated_labels = torch.max(outputs, 1)[1]
                            targets = targets.to(rank)
                            num_examples += targets.size(0)
                            mae += torch.sum(torch.abs(estimated_labels - targets))
                            mse += torch.sum((estimated_labels - targets)**2)
                        mae = mae.float() / num_examples
                        mse = mse.float() / num_examples

                    print('Iteration '+ str(iteration))
                    eval_text = 'TEST %s' % 'MAE/RMSE: %.2f/%.2f' % (mae, torch.sqrt(mse))
                    print(eval_text)
                    time_text = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
                    print(time_text)

                    with open(LOGFILE, 'a') as f:
                        f.write('%s\n' % eval_text)
                        f.write('%s\n' % time_text)

                    # tensorboard
                    writer.add_scalar('MAE/test', mae, counter)
                    writer.add_scalar('MSE/test', torch.sqrt(mse), counter)

                    if mae < best_mae:
                        break_str = 'Break the best score Iteration ' + str(iteration)
                        print(break_str)
                        with open(LOGFILE, 'a') as f:
                            f.write('%s\n' % (break_str))

                        best_mae = mae
                        if checkpoint is not None:
                            os.remove(checkpoint)
                        checkpoint = os.path.join(PATH, 'iteration_{}.pt'.format(iteration))
                        torch.save({
                                'iteration' : iteration,
                                'model_state_dict' : ddp_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': cost
                    }, checkpoint)

                    mae, mse, num_examples = 0, 0, 0
                    counter += 1

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)
