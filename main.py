import torch
import random
import numpy as np
from configs import args as Arg
import json
import random
import sys
import time
from torch import nn
from transforms import ScanpathTransform
from data import read_dataset, FixationDataset
from models.ffn import FFNGenerator
from models.ssl import BarlowTwins, LARS, adjust_learning_rate

def main():
    args = Arg()
    args.ngpus_per_node = torch.cuda.device_count()
    # single-node distributed training
    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = args.ngpus_per_node
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    
def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    # Model
    backbone = FFNGenerator((args.img_height, args.img_width), args.ffn_size)
    model = BarlowTwins(args, backbone).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    
    # Optimizer
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0
    
    # Dataset
    anno, _ = read_dataset(args.data_dir, args.datasets, val_ratio=0.0)
    dataset = FixationDataset(anno, ScanpathTransform(**args.transform))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, collate_fn = dataset.collate_function, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):
            y1 = [y.cuda(gpu, non_blocking=True) for y in y1]
            y2 = [y.cuda(gpu, non_blocking=True) for y in y2]
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet50.pth')
        
if __name__ == '__main__':    
    main()