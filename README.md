# Introduction
PyTorch code for the ICLR 2021 paper [[i-Mix: A Domain-Agnostic Strategy for Contrastive Representation Learning](https://arxiv.org/abs/2010.08887)].
```
@inproceedings{lee2021imix,
  title={i-Mix: A Domain-Agnostic Strategy for Contrastive Representation Learning},
  author={Lee, Kibok and Zhu, Yian and Sohn, Kihyuk and Li, Chun-Liang and Shin, Jinwoo and Lee, Honglak},
  booktitle={ICLR},
  year={2021}
}
```

# Dependencies
- python 3.7.4
- numpy 1.17.2
- pytorch 1.4.0
- torchvision 0.5.0
- cudatoolkit 10.1
- librosa 0.8.0 for `speech_commands`
- PIL 6.2.0 for `GaussianBlur`

# Data
- CIFAR-10/100 will automatically be downloaded.
- For ImageNet, please refer to the [[PyTorch ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet)]. The folder structure should be like `data/imagenet/train/n01440764/`
- For speech commands, run `bash speech_commands/download_speech_commands_dataset.sh`
- For tabular datasets, download [[covtype.data.gz](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz)] and place it to `data/` and [[HIGGS.csv.gz](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz)] and place them in `data/`. They are processed when first loaded.

# Running scripts
Please refer to [[run.sh](run.sh)].


For those who want to apply our method in their own code, we provide a minimal example based on [[MoCo](https://github.com/facebookresearch/moco)]:
```
# mixup: somewhere in main_moco.py
def mixup(input, alpha):
    beta = torch.distributions.beta.Beta(alpha, alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    lam = beta.sample([input.shape[0]]).to(device=input.device)
    lam = torch.max(lam, 1. - lam)
    lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
    output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return output, randind, lam

# cutmix: somewhere in main_moco.py
def cutmix(input, alpha):
    beta = torch.distributions.beta.Beta(alpha, alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    lam = beta.sample().to(device=input.device)
    lam = torch.max(lam, 1. - lam)
    (bbx1, bby1, bbx2, bby2), lam = rand_bbox(input.shape[-2:], lam)
    output = input.clone()
    output[..., bbx1:bbx2, bby1:bby2] = output[randind][..., bbx1:bbx2, bby1:bby2]
    return output, randind, lam

def rand_bbox(size, lam):
    W, H = size
    cut_rat = (1. - lam).sqrt()
    cut_w = (W * cut_rat).to(torch.long)
    cut_h = (H * cut_rat).to(torch.long)

    cx = torch.zeros_like(cut_w, dtype=cut_w.dtype).random_(0, W)
    cy = torch.zeros_like(cut_h, dtype=cut_h.dtype).random_(0, H)

    bbx1 = (cx - cut_w // 2).clamp(0, W)
    bby1 = (cy - cut_h // 2).clamp(0, H)
    bbx2 = (cx + cut_w // 2).clamp(0, W)
    bby2 = (cy + cut_h // 2).clamp(0, H)

    new_lam = 1. - (bbx2 - bbx1).to(lam.dtype) * (bby2 - bby1).to(lam.dtype) / (W * H)

    return (bbx1, bby1, bbx2, bby2), new_lam

# https://github.com/facebookresearch/moco/blob/master/main_moco.py#L193
criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)

# https://github.com/facebookresearch/moco/blob/master/main_moco.py#L302-L303
images[0], target_aux, lam = mixup(images[0], alpha=1.)
# images[0], target_aux, lam = cutmix(images[0], alpha=1.)
target = torch.arange(images[0].shape[0], dtype=torch.long).cuda()
output, _ = model(im_q=images[0], im_k=images[1])
loss = lam * criterion(output, target) + (1. - lam) * criterion(output, target_aux)

# https://github.com/facebookresearch/moco/blob/master/moco/builder.py#L142-L149
contrast = torch.cat([k, self.queue.clone().detach().t()], dim=0)
logits = torch.mm(q, contrast.t())
```


# Note
- `builder.py` is adapted from [[MoCo](https://github.com/facebookresearch/moco/blob/master/moco/builder.py)] and [[PyContrast](https://github.com/HobbitLong/PyContrast/tree/master/pycontrast/learning)].
- `main_*.py` is adapted from [[PyTorch ImageNet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py)] and [[Mo](https://github.com/facebookresearch/moco/blob/master/main_moco.py)[Co](https://github.com/facebookresearch/moco/blob/master/main_lincls.py)].
- `models/resnet.py` is adapted from [[torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)].
- `speech_commands/` is adapted from [[this repo](https://github.com/tugstugi/pytorch-speech-commands)].
