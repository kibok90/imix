#!/usr/bin/env python
import argparse
import builtins
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

import models
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
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=10., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lrd', '--learning-rate-decay', default=0.2, type=float,
                    metavar='LRD', help='learning rate decay', dest='lrd')
parser.add_argument('--schedule', default=[80, 90, 95], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
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

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--tb', action='store_true',
                    help='tensorboard')
parser.add_argument('--start-eval', default=0, type=int,
                    help='epoch number when starting evaluation')
parser.add_argument('--trial', default=None, type=str,
                    help='auxiliary string to distinguish trials')
parser.add_argument('--finetune', action='store_true',
                    help='do not freeze CNN')
parser.add_argument('--new-resume', action='store_true',
                    help='reset optimizer and start_epoch')
parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet', 'imagenet_val', 'cifar10', 'cifar100', 'speech_commands', 'covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall'],
                    help='dataset to use')
parser.add_argument('--class-ratio', default=1., type=float,
                    help='reduce training dataset size by removing classes')
parser.add_argument('--data-ratio', default=1., type=float,
                    help='reduce training dataset size by removing data per class')

best_acc1 = 0


def main(args):

    # scale lr
    args.lr = args.lr * args.batch_size / 256
    print('lr is scaled to {}'.format(args.lr))

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
    global best_acc1
    best_acc1 = 0
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
    in_channels = 3
    if 'imagenet' in args.dataset:
        num_classes = 1000
    elif args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'speech_commands':
        num_classes = len(speech_commands.CLASSES)
        in_channels = 1
    elif args.dataset in ['covtype']:
        num_classes = 7
        in_channels = [10, 4, 40]
    elif args.dataset in ['higgs', 'higgs100K', 'higgs1M']:
        num_classes = 2
        in_channels = 21
    elif args.dataset in ['higgsall', 'higgs100Kall', 'higgs1Mall']:
        num_classes = 2
        in_channels = 28
    num_classes = int(num_classes * args.class_ratio)
    small = 'imagenet' not in args.dataset

    model = models.__dict__[args.arch](num_classes=num_classes, in_channels=in_channels, small=small)

    if not args.finetune:
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q.') and not k.startswith('module.encoder_q.fc.'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                elif k.startswith('encoder_q.') and not k.startswith('encoder_q.fc.'):
                    # remove prefix
                    state_dict[k[len("encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, set(msg.missing_keys)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    else:
        print("=> no pre-trained model")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
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
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            # model.features = torch.nn.DataParallel(model.features)
            # model.cuda()
        # else:
            # model = torch.nn.DataParallel(model).cuda()
        model.cuda()
        # apply DataParallel after resume

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if not args.finetune:
        assert len(parameters) == 2, len(parameters) # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if '.pth' not in args.resume: # fr
            args.resume = os.path.join(os.path.dirname(args.pretrained), 'checkpoint_lincls_{}_{}{}.pth'.format(args.dataset, 'ft' if args.resume == 'ft' else 'fr', '' if args.trial is None else '_{}'.format(args.trial)))

        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            best_acc1 = checkpoint['best_acc1']
            if (args.gpu is not None) and isinstance(best_acc1, torch.Tensor):
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            if args.distributed:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            if not args.new_resume:
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # check if training is already done
    if args.start_epoch >= args.epochs:
        if not args.evaluate:
            return
    elif args.tb and not (args.multiprocessing_distributed and args.gpu != 0):
        from torch.utils.tensorboard import SummaryWriter
        tb_logger = SummaryWriter(log_dir=os.path.join(os.path.dirname(args.pretrained), 'tb_lincls_{}{}'.format(args.dataset, '' if args.trial is None else '_{}'.format(args.trial))))

    # apply DataParallel after resume
    if (not args.distributed) and (args.gpu is None):
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
        else:
            model = torch.nn.DataParallel(model)

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
    elif args.dataset == 'imagenet_val':
        traindir = valdir = os.path.join(args.data, 'val')
    elif args.dataset == 'speech_commands':
        traindir = os.path.join(args.data, 'speech_commands/train')
        valdir = os.path.join(args.data, 'speech_commands/test')
    else:
        traindir = valdir = args.data
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
        train_transform = None
    elif 'imagenet' in args.dataset:
        train_transform = \
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    elif args.dataset == 'speech_commands':
        train_transform = transforms.Compose(
            [speech_commands.LoadAudio(), speech_commands.FixAudioLength(),
             speech_commands.ToMelSpectrogram(n_mels=32), speech_commands.ToTensor('mel_spectrogram')])
    else:
        train_transform = \
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
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
        train_loader = tabular.TabularDataLoader(data=train_db[0], targets=train_db[1], shuffle=True, drop_last=True, num_copies=1)
        val_loader = tabular.TabularDataLoader(data=test_db[0], targets=test_db[1], shuffle=False, drop_last=False, num_copies=1)
    elif args.dataset in ['higgs', 'higgsall']:
        train_db, test_db = tabular.higgs(root=traindir, mode='all', channels=in_channels, do_normalize=True)
        train_loader = tabular.TabularDataLoader(data=train_db[0], targets=train_db[1], shuffle=True, drop_last=True, num_copies=1)
        val_loader = tabular.TabularDataLoader(data=test_db[0], targets=test_db[1], shuffle=False, drop_last=False, num_copies=1)
    elif args.dataset in ['higgs100K', 'higgs100Kall']:
        train_db, test_db = tabular.higgs(root=traindir, mode='100k', channels=in_channels, do_normalize=True)
        train_loader = tabular.TabularDataLoader(data=train_db[0], targets=train_db[1], shuffle=True, drop_last=True, num_copies=1)
        val_loader = tabular.TabularDataLoader(data=test_db[0], targets=test_db[1], shuffle=False, drop_last=False, num_copies=1)
    elif args.dataset in ['higgs1M', 'higgs1Mall']:
        train_db, test_db = tabular.higgs(root=traindir, mode='1M', channels=in_channels, do_normalize=True)
        train_loader = tabular.TabularDataLoader(data=train_db[0], targets=train_db[1], shuffle=True, drop_last=True, num_copies=1)
        val_loader = tabular.TabularDataLoader(data=test_db[0], targets=test_db[1], shuffle=False, drop_last=False, num_copies=1)
    else:
        raise NotImplementedError('unsupported dataset: {}'.format(args.dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.dataset not in ['covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if 'imagenet' in args.dataset:
        val_transform = \
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    elif args.dataset == 'speech_commands':
        val_transform = transforms.Compose(
            [speech_commands.LoadAudio(), speech_commands.FixAudioLength(),
             speech_commands.ToMelSpectrogram(n_mels=32), speech_commands.ToTensor('mel_spectrogram')])
    else:
        val_transform = \
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    if args.dataset in ['imagenet', 'imagenet_val']:
        print('ImageNet ImageFolder at: {}'.format(traindir))
        # val_dataset = datasets.ImageFolder(valdir, val_transform)
        import folder
        val_dataset = folder.ImageFolder(valdir, val_transform, class_ratio=args.class_ratio, data_ratio=1.)
    elif args.dataset == 'cifar10':
        print('CIFAR-10 at: {}'.format(valdir))
        val_dataset = datasets.CIFAR10(root=valdir, train=False, transform=val_transform, download=False)
    elif args.dataset == 'cifar100':
        print('CIFAR-100 at: {}'.format(valdir))
        val_dataset = datasets.CIFAR100(root=valdir, train=False, transform=val_transform, download=False)
    elif args.dataset == 'speech_commands':
        print('Speech Commands at: {}'.format(valdir))
        val_dataset = speech_commands.SpeechCommandsDataset(valdir, train_transform, silence_percentage=0)
    elif args.dataset in ['covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
        pass
    else:
        raise NotImplementedError('unsupported dataset: {}'.format(args.dataset))

    if args.dataset not in ['covtype', 'higgs', 'higgsall', 'higgs100K', 'higgs100Kall', 'higgs1M', 'higgs1Mall']:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        acc1 = validate(val_loader, model, criterion, args.start_epoch, None, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, tb_logger, args)

        if (epoch >= args.start_eval) or (epoch == args.epochs-1):
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, epoch, tb_logger, args)
        else:
            acc1 = 0

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

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
            is_milestone = (epoch + 1 == args.schedule[0]) and 'imagenet' in args.dataset
            if epoch == args.epochs-1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': save_state_dict,
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, is_milestone=False, filename='{}/checkpoint_lincls_{}_{}{}.pth'.format(os.path.dirname(args.pretrained), args.dataset, 'ft' if args.finetune else 'fr', '' if args.trial is None else '_{}'.format(args.trial)), dataset=args.dataset, epoch=epoch+1)
            if (epoch == args.start_epoch) and (not args.finetune) and (args.pretrained):
                sanity_check(model.state_dict(), args.pretrained)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        with open('log.txt', 'a') as f:
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(args.pretrained, args.dataset, args.trial, 'ft' if args.finetune else 'fr', acc1))


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

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if args.finetune:
        model.train()
    else:
        model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.long().cuda(args.gpu, non_blocking=True)
        bsz = target.size(0)

        # compute output
        output = model(images)
        loss = criterion(output, target)

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


def validate(val_loader, model, criterion, epoch, tb_logger, args):
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
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
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

    return top1.avg


def save_checkpoint(state, is_best, is_milestone, filename='checkpoint.pth', dataset='dataset', epoch=-1):
    torch.save(state, filename)
    # if is_best:
        # shutil.copyfile(filename, filename.replace('checkpoint_lincls', 'model_best'))
    if is_milestone:
        shutil.copyfile(filename, os.path.splitext(filename)[0] + '_{:d}.pth'.format(epoch))


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        if k_pre not in state_dict_pre:
            k_pre = k_pre[len('module.'):]

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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
    for milestone in args.schedule:
        lr *= args.lrd if epoch >= milestone else 1.
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


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
