PyTorch code for the ICLR 2021 paper [[i-Mix: A Domain-Agnostic Strategy for Contrastive Representation Learning](https://arxiv.org/abs/2010.08887)].

Code coming soon.

In the meantime, if you want a practical example of our method, here is a quick implementation based on MoCo:

```
# somewhere in main_moco.py
def mixup(input, alpha):
    beta = torch.distributions.beta.Beta(alpha, alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    lam = beta.sample([input.shape[0]]).to(device=input.device)
    lam = torch.max(lam, 1. - lam)
    lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
    output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return output, randind, lam

# https://github.com/facebookresearch/moco/blob/master/main_moco.py#L193
criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)

# https://github.com/facebookresearch/moco/blob/master/main_moco.py#L302-L303
images[0], target_aux, lam = mixup(images[0], alpha=1.)
target = torch.arange(images[0].shape[0], dtype=torch.long).cuda()
output, _ = model(im_q=images[0], im_k=images[1])
loss = lam * criterion(output, target) + (1. - lam) * criterion(output, target_aux)

# https://github.com/facebookresearch/moco/blob/master/moco/builder.py#L142-L149
contrast = torch.cat([k, self.queue.clone().detach().t()], dim=0)
logits = torch.mm(q, contrast.t())
```
