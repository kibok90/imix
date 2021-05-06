#!/usr/bin/env python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models

from PIL import ImageFilter # for GaussianBlur
from builder import Builder # contrastive learner
import models # backbone
import speech_commands
import tabular


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.125, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lrd', '--learning-rate-decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay', dest='lrd')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10000', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--qlen', default=4096, type=int,
                    help='queue size; number of negative keys (default: 4096)')
parser.add_argument('--emam', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temp', default=0.2, type=float,
                    help='softmax temperature (default: 0.2)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--tb', action='store_true',
                    help='tensorboard')
parser.add_argument('--save', default='save', type=str,
                    help='save root')
parser.add_argument('--save-freq', default=100, type=int,
                    help='save checkpoint frequency')
parser.add_argument('--trial', default=None, type=str,
                    help='auxiliary string to distinguish trials')
parser.add_argument('--warm', action='store_true',
                    help='use learning rate warm-up')
parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet', 'imagenet_val', 'cifar10', 'cifar100', 'speech_commands', 'covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall'],
                    help='dataset to use')
parser.add_argument('--proj', default='mlpbn1', type=str, choices=['lin', 'linbn', 'mlp', 'mlpbn', 'mlpbn1'],
                    help='projection layer')
parser.add_argument('--pred', default='none', type=str, choices=['none', 'lin', 'linbn', 'mlp', 'mlpbn', 'mlpbn1'],
                    help='prediction layer')
parser.add_argument('--method', default='moco', type=str, choices=['npair', 'moco', 'mocon', 'byol'],
                    help='method to use')
parser.add_argument('--imix', default='none', type=str, choices=['none', 'imixup', 'icutmix'],
                    help='i-mix')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='i-mix hyperparameter')
parser.add_argument('--inputmix', default=0, type=int,
                    help='number of auxiliary data for inputmix')
parser.add_argument('--alpha2', default=1.0, type=float,
                    help='inputmix hyperparameter')
parser.add_argument('--lincls', action='store_true',
                    help='run lincls after training')
parser.add_argument('--lincls-more', action='store_true',
                    help='use similar datasets in eval')
parser.add_argument('--cos-m', action='store_true',
                    help='use cosine schedule for moco momentum of updating key encoder')
parser.add_argument('--head-mul', default=1, type=int,
                    help='increase projection/prediction head size')
parser.add_argument('--sym', action='store_true',
                    help='symmetrized loss')
parser.add_argument('--bn', default='shuffle', type=str, choices=['none', 'shuffle', 'sync'],
                    help='shuffleBN or syncBN or not')
parser.add_argument('--class-ratio', default=1., type=float,
                    help='reduce training dataset size by removing classes')
parser.add_argument('--data-ratio', default=1., type=float,
                    help='reduce training dataset size by removing data per class')
parser.add_argument('--eval-class-ratio', default=1., type=float,
                    help='reduce downstream dataset size by removing classes')
parser.add_argument('--eval-data-ratio', default=1., type=float,
                    help='reduce downstream dataset size by removing data per class')
parser.add_argument('--mix-layers', type=str, default='0', help='manifold mixup')
parser.add_argument('--no-aug', action='store_true',
                    help='no augmentation')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam'],
                    help='optimizer')
parser.add_argument('--inputdrop', default=0.0, type=float,
                    help='input drop rate')
parser.add_argument('--pinv', action='store_true',
                    help='pinv')

def main(args):

    # alpha = 0 when no i-mix
    if args.imix == 'none':
        args.alpha = 0.
    # include non-positive im_k as negs when i-mix for moco
    elif args.method == 'moco':
        args.method = 'mocon'
    # alpha2 = 0 when no inputmix
    if args.inputmix == 0:
        args.alpha2 = 0.
    # no shuffleBN for npair
    if (args.method == 'npair') and args.bn == 'shuffle':
        args.bn = 'none'
    # no temperature scaling for byol
    if args.method == 'byol':
        args.t = 1.

    # model name
    method_str = args.method
    if args.sym:
        method_str = f'{method_str}_sym'
    if args.cos_m:
        method_str = f'{method_str}_cosm'
    if args.head_mul > 1:
        method_str = f'{method_str}_hm{args.head_mul}'
    if args.method in ['moco', 'mocon']:
        method_str = f'{method_str}_k{args.qlen}'
    if args.method in ['moco', 'mocon', 'byol']:
        method_str = f'{method_str}_m{args.emam}'
    if args.method in ['npair', 'moco', 'mocon']:
        method_str = f'{method_str}_t{args.temp}'
    method_str = f'{method_str}_proj_{args.proj}'
    if args.pred != 'none':
        method_str = f'{method_str}_pred_{args.pred}'
    if args.bn != 'none':
        method_str = f'{method_str}_{args.bn}bn'
    if args.no_aug:
        method_str = f'{method_str}_noaug'
    if args.inputdrop > 0.:
        method_str = f'{method_str}_indrop{args.inputdrop}'
    inputmix_str = '' if args.inputmix == 0 else f'{args.inputmix}inputmix{args.alpha2}'
    mix_layers_str = '' if args.mix_layers == '0' else args.mix_layers
    args.model_name = f'{args.save}/{args.dataset}/{args.arch}_{method_str}_{inputmix_str}{mix_layers_str}{args.imix}{args.alpha}_lr{args.lr}_wd{args.weight_decay:.1e}_bsz{args.batch_size}_ep{args.epochs}'
    if args.class_ratio < 1.:
        args.model_name = f'{args.model_name}_cr{args.class_ratio}'
    if args.data_ratio < 1.:
        args.model_name = f'{args.model_name}_dr{args.data_ratio}'

    # scale lr
    args.lr = args.lr * args.batch_size / 256
    print('lr is scaled to {}'.format(args.lr))

    # warm-up for large-batch training
    if args.batch_size > 256:
        args.warm = True
    if args.optim != 'sgd':
        args.model_name = f'{args.model_name}_{args.optim}'
    if args.cos:
        args.model_name = f'{args.model_name}_cos'
    if args.warm:
        args.model_name = f'{args.model_name}_warm'
        args.warmup_from = 0.
        if args.epochs > 500:
            args.warm_epochs = 10
        else:
            args.warm_epochs = 5
        if args.cos:
            eta_min = args.lr * 0.001
            args.warmup_to = eta_min + (args.lr - eta_min) * (
                        1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.lr

    if args.trial is not None:
        args.model_name = f'{args.model_name}_trial{args.trial}'

    if not os.path.isdir(args.model_name):
        os.makedirs(args.model_name)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    tb_logger = None

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset in ['speech_commands']:
        in_channels = 1
    elif args.dataset in ['covtype']:
        in_channels = [10, 4, 40]
    elif args.dataset in ['higgs', 'higgs100K', 'higgs1M']:
        in_channels = 21
    elif args.dataset in ['higgsall', 'higgs100Kall', 'higgs1Mall']:
        in_channels = 28
    else:
        in_channels = 3
    args.in_channels = in_channels
    small = 'imagenet' not in args.dataset
    model = Builder(
        models.__dict__[args.arch],
        args.dim, args.qlen, args.emam, args.temp, args.proj,
        pred=args.pred, method=args.method, shuffle_bn=(args.bn == 'shuffle'), head_mul=args.head_mul, sym=args.sym, in_channels=in_channels, small=small, distributed=args.distributed)
    # print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if args.bn == 'sync':
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        if args.bn == 'sync':
            import sync_batchnorm
            model = sync_batchnorm.convert_model(model)
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
        model.cuda()
        # apply DataParallel after resume

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if '.pth' not in args.resume: # find the last model
            resume_dir = args.resume if os.path.isdir(args.resume) else args.model_name
            args.resume = os.path.join(resume_dir, 'checkpoint.pth')

        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            if args.distributed:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # check if training is already done
    if (args.start_epoch >= args.epochs) and not args.pinv:
        return
    elif args.tb and not (args.multiprocessing_distributed and args.gpu != 0):
        from torch.utils.tensorboard import SummaryWriter
        tb_logger = SummaryWriter(log_dir=os.path.join(args.model_name, 'tb_pretext'))

    # apply DataParallel after resume
    if (not args.distributed) and (args.gpu is None):
        model.encoder_q = torch.nn.DataParallel(model.encoder_q)
        if model.encoder_k is not None:
            model.encoder_k = torch.nn.DataParallel(model.encoder_k)
        if model.pred is not None:
            model.pred = torch.nn.DataParallel(model.pred)

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
    elif args.dataset == 'imagenet_val':
        traindir = os.path.join(args.data, 'val')
    elif args.dataset == 'speech_commands':
        traindir = os.path.join(args.data, 'speech_commands/train')
    else:
        traindir = args.data
    if 'imagenet' in args.dataset:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    elif args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2009, 0.1984, 0.2023])
    else:
        normalize = None

    if args.dataset in ['covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
        train_transform_list = None
    elif args.no_aug:
        if args.dataset == 'speech_commands':
            train_transform_list = [
                speech_commands.LoadAudio(),
                speech_commands.FixAudioLength(),
                speech_commands.ToMelSpectrogram(n_mels=32),
                speech_commands.ToTensor('mel_spectrogram')
            ]
        else:
            train_transform_list = [
                transforms.ToTensor(),
                normalize,
            ]
    elif 'imagenet' in args.dataset:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        train_transform_list = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif args.dataset == 'speech_commands':
        data_aug_transform = transforms.Compose(
            [speech_commands.ChangeAmplitude(), speech_commands.ChangeSpeedAndPitchAudio(), speech_commands.FixAudioLength(),
             speech_commands.ToSTFT(), speech_commands.StretchAudioOnSTFT(),
             speech_commands.TimeshiftAudioOnSTFT(), speech_commands.FixSTFTDimension()])
        background_noise_dir = os.path.join(traindir, '_background_noise_')
        bg_dataset = speech_commands.BackgroundNoiseDataset(background_noise_dir, data_aug_transform)
        train_transform_list = [
            speech_commands.LoadAudio(),
            data_aug_transform,
            speech_commands.AddBackgroundNoiseOnSTFT(bg_dataset),
            speech_commands.ToMelSpectrogramFromSTFT(n_mels=32), speech_commands.DeleteSTFT(),
            speech_commands.ToTensor('mel_spectrogram')
        ]
    else:
        # SimCLR's aug for CIFAR: color distortion by (0.8, 0.8, 0.8, 0.2) * 0.5, no GaussianBlur (ref: Appendix B.9 Setup in SimCLR)
        train_transform_list = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    if args.dataset in ['covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
        train_transform = TwoCropsTransform(None)
    else:
        train_transform = TwoCropsTransform(transforms.Compose(train_transform_list))
    if args.dataset in ['imagenet', 'imagenet_val']:
        print('ImageNet ImageFolder at: {}'.format(traindir))
        # train_dataset = datasets.ImageFolder(traindir, train_transform)
        import folder
        train_dataset = folder.ImageFolder(traindir, train_transform, class_ratio=args.class_ratio, data_ratio=args.data_ratio)
    elif args.dataset == 'cifar10':
        print('CIFAR-10 at: {}'.format(traindir))
        train_dataset = datasets.CIFAR10(root=traindir, train=True, transform=train_transform, download=True)
    elif args.dataset == 'cifar100':
        print('CIFAR-100 at: {}'.format(traindir))
        train_dataset = datasets.CIFAR100(root=traindir, train=True, transform=train_transform, download=True)
    elif args.dataset == 'speech_commands':
        print('Speech Commands at: {}'.format(traindir))
        train_dataset = speech_commands.SpeechCommandsDataset(traindir, train_transform)
    elif args.dataset == 'covtype':
        train_db, test_db = tabular.covtype(root=traindir, channels=in_channels, do_normalize=True)
        train_loader = tabular.TabularDataLoader(data=train_db[0], targets=train_db[1], shuffle=True, drop_last=True, num_copies=2)
    elif args.dataset in ['higgs', 'higgsall']:
        train_db, test_db = tabular.higgs(root=traindir, mode='all', channels=in_channels, do_normalize=True)
        train_loader = tabular.TabularDataLoader(data=train_db[0], targets=train_db[1], shuffle=True, drop_last=True, num_copies=2)
    elif args.dataset in ['higgs100K', 'higgs100Kall']:
        train_db, test_db = tabular.higgs(root=traindir, mode='100k', channels=in_channels, do_normalize=True)
        train_loader = tabular.TabularDataLoader(data=train_db[0], targets=train_db[1], shuffle=True, drop_last=True, num_copies=2)
    elif args.dataset in ['higgs1M', 'higgs1Mall']:
        train_db, test_db = tabular.higgs(root=traindir, mode='1M', channels=in_channels, do_normalize=True)
        train_loader = tabular.TabularDataLoader(data=train_db[0], targets=train_db[1], shuffle=True, drop_last=True, num_copies=2)
    else:
        raise NotImplementedError('unsupported dataset: {}'.format(args.dataset))

    if args.dataset not in ['covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, tb_logger, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            model_state_dict = model.state_dict()
            save_state_dict = collections.OrderedDict()
            for key in model.state_dict():
                if 'module.' in key:
                    pos = key.find('module.')
                    new_key = key[:pos] + key[pos+len('module.'):]
                else:
                    new_key = key
                save_state_dict.update({new_key: model_state_dict[key]})
            is_milestone = (args.save_freq > 0 and ((epoch + 1) % args.save_freq == 0)) or (epoch == args.epochs - 1)
            if is_milestone:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': save_state_dict,
                    'optimizer' : optimizer.state_dict(),
                }, is_milestone=False, filename='{}/checkpoint.pth'.format(args.model_name), epoch=epoch+1)

    # pinv eval
    if args.pinv:
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if args.dataset in ['covtype']:
                args.num_classes = 7
            elif args.dataset in ['higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
                args.num_classes = 2
            else:
                raise NotImplementedError('pinv is for tabular only')
            pinv_train_loader = tabular.TabularDataLoader(data=train_db[0], targets=train_db[1], shuffle=True, drop_last=True, num_copies=1)
            val_loader = tabular.TabularDataLoader(data=test_db[0], targets=test_db[1], shuffle=False, drop_last=False, num_copies=1)
            if hasattr(model, 'module'):
                this_model = model.module.encoder_q
            # elif hasattr(model.encoder_q, 'module'):
            #     this_model = model.encoder_q.module
            else:
                this_model = model.encoder_q
            pinv_features, pinv_targets = get_stat(pinv_train_loader, this_model, args)
            classifier = pinv_solve(pinv_features, pinv_targets, args)
            pinv_loss, pinv_top1 = validate(val_loader, this_model, classifier, criterion, args.epochs, None, args)
            with open('log.txt', 'a') as f:
                f.write('{}\t{}\t{}\t{}\t{}\n'.format(args.model_name, args.dataset, args.trial, 'pinv', pinv_top1))


def train(train_loader, model, criterion, optimizer, epoch, tb_logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.warm and epoch < args.warm_epochs:
            warmup_learning_rate(optimizer, epoch, i, len(train_loader), args)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)
        if args.inputdrop > 0.:
            if isinstance(args.in_channels, (tuple, list)):
                ch = args.in_channels[0]
                images[0][:,:ch] = images[0][:,:ch] * (torch.rand_like(images[0][:,:ch]) > args.inputdrop).float()
                images[1][:,:ch] = images[1][:,:ch] * (torch.rand_like(images[1][:,:ch]) > args.inputdrop).float()
            else:
                images[0] = images[0] * (torch.rand_like(images[0]) > args.inputdrop).float()
                images[1] = images[1] * (torch.rand_like(images[1]) > args.inputdrop).float()
        bsz = images[0].size(0)

        # compute output
        if args.cos_m:
            p = (i + 1 + epoch * len(train_loader)) / (args.epochs * len(train_loader))
            m = 1 - (1 - args.m) * (1 + math.cos(math.pi * p)) / 2
        else:
            m = None
        output, target, loss = model(im_1=images[0], im_2=images[1], m=m, criterion=criterion, imix=args.imix, alpha=args.alpha, mix_layers=args.mix_layers, num_aux=args.inputmix, alpha2=args.alpha2)
        # loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), bsz)
        top1.update(acc1[0], bsz)
        top5.update(acc5[0], bsz)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (i == len(train_loader) - 1):
            progress.display(i)

    # tensorboard
    if tb_logger is not None:
        tb_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        tb_logger.add_scalar('train/loss', losses.avg, global_step=epoch)
        tb_logger.add_scalar('train/top1', top1.avg, global_step=epoch)
        tb_logger.add_scalar('train/top5', top5.avg, global_step=epoch)


def save_checkpoint(state, is_milestone, filename='checkpoint.pth', epoch=-1):
    torch.save(state, filename)
    if is_milestone:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'checkpoint_{:d}.pth'.format(epoch)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= args.lrd if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, epoch, batch_id, total_batches, args):
    p = (batch_id + 1 + epoch * total_batches) / (args.warm_epochs * total_batches)
    lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_stat(train_loader, model, args):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    end = time.time()
    pinv_features = 0
    pinv_targets = 0

    with torch.no_grad():
        for idx, (images, targets) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True).long()

            # extract features
            features = model(images, no_fc=True)
            target_matrix = torch.zeros([features.shape[0], args.num_classes], dtype=features.dtype, device=features.device)
            target_matrix.scatter_(1, targets[:, None], 1)
            pinv_features += features.t() @ features
            pinv_targets += features.t() @ target_matrix

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return pinv_features, pinv_targets


def pinv_solve(features, targets, args):
    weight = targets.t() @ features.pinverse()

    classifier = torch.nn.Linear(features.shape[1], args.num_classes, bias=False)
    classifier = classifier.cuda()
    classifier.weight.data = weight

    return classifier


def validate(val_loader, model, classifier, criterion, epoch, tb_logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.long().cuda(args.gpu, non_blocking=True)
            bsz = target.size(0)

            # compute output
            features = model(images, no_fc=True)
            output = classifier(features)
            loss = criterion(output, target).mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), bsz)
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % args.print_freq == 0) or (i == len(val_loader) - 1):
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    # tensorboard
    if tb_logger is not None:
        tb_logger.add_scalar('val/loss', losses.avg, global_step=epoch)
        tb_logger.add_scalar('val/top1', top1.avg, global_step=epoch)
        tb_logger.add_scalar('val/top5', top5.avg, global_step=epoch)

    return losses.avg, top1.avg


class TwoCropsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        if self.transform is None:
            return x, x
        else:
            q = self.transform(x)
            k = self.transform(x)
            return [q, k]


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    main(args)

    # lincls
    if args.lincls:
        import main_lincls
        target_datasets = [args.dataset]
        if args.lincls_more:
            if args.dataset == 'cifar10':
                target_datasets.append('cifar100')
            elif args.dataset == 'cifar100':
                target_datasets.append('cifar10')
        if 'imagenet' in args.dataset:
            lin_lrs = ['30']
            lin_schedule = ['--lrd', '0.1', '--schedule', '60', '80', '--epochs', '100']
            aux = ''
        else:
            lin_lrs = ['10', '30', '50', '70', '1', '3', '5']
            lin_schedule = ['--lrd', '0.2', '--schedule', '80', '90', '95', '--epochs', '100']
            aux = '_8990'
        for target_dataset in target_datasets:
            for lin_lr in lin_lrs:
                trial_str = 'lr{}{}'.format(lin_lr, aux)
                if args.eval_data_ratio < 1.:
                    trial_str = '{}_dr{}'.format(trial_str, args.eval_data_ratio)
                if args.eval_class_ratio < 1.:
                    trial_str = '{}_cr{}'.format(trial_str, args.eval_class_ratio)
                args_lincls_list = [args.data, '-a', args.arch, '-j', str(args.workers), '--dataset', target_dataset, '--start-eval', '80', '--lr', lin_lr, '--trial', trial_str]
                args_lincls_list.extend(lin_schedule)
                if args.eval_data_ratio < 1.:
                    args_lincls_list.extend(['--data-ratio', str(args.eval_data_ratio)])
                if args.eval_class_ratio < 1.:
                    args_lincls_list.extend(['--class-ratio', str(args.eval_class_ratio)])
                if args.tb:
                    args_lincls_list.append('--tb')
                if args.resume:
                    args_lincls_list.extend(['--resume', 'true'])
                if args.multiprocessing_distributed:
                    args_lincls_list.extend(['--multiprocessing-distributed', '--dist-url', args.dist_url])
                args_lincls = main_lincls.parser.parse_args(args_lincls_list)
                args_lincls.pretrained = os.path.join(args.model_name, 'checkpoint.pth')
                print(args_lincls)

                main_lincls.main(args_lincls)

