import torch
import torch.nn as nn

class Builder(nn.Module):

    def __init__(self, base_encoder, dim=128, qlen=4096, emam=0.999, temp=0.2, proj='lin', pred='none', method='npair', shuffle_bn=False, head_mul=1, sym=False, in_channels=3, small=False, distributed=False, kaiming_init=True):
        super(Builder, self).__init__()

        self.qlen = qlen
        self.emam = emam
        self.temp = temp
        self.method = method
        self.shuffle_bn = shuffle_bn
        self.sym = sym
        self.distributed = distributed

        # encoder
        self.encoder_q = base_encoder(num_classes=dim*head_mul, in_channels=in_channels, small=small, kaiming_init=kaiming_init)
        self.encoder_k = base_encoder(num_classes=dim*head_mul, in_channels=in_channels, small=small, kaiming_init=kaiming_init)

        # projection head
        dim_out, dim_in = self.encoder_q.fc.weight.shape
        dim_mlp = dim_in * head_mul
        if proj == 'mlpbn':
            print('MLP projection layer with BN')
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out), BatchNorm1d(dim_out))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out), BatchNorm1d(dim_out))
        elif proj == 'mlpbn1':
            print('MLP projection layer with BN in the middle')
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
        elif proj == 'mlp':
            print('MLP projection layer without BN')
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
        elif proj == 'linbn':
            print('Linear projection layer with BN')
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_in, dim_out), BatchNorm1d(dim_out))
        else:
            print('Linear projection layer without BN')

        # prediction head 
        if pred == 'mlpbn':
            print('MLP prediction layer with BN')
            self.pred = nn.Sequential(nn.Linear(dim_out, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out), BatchNorm1d(dim_out))
        elif pred == 'mlpbn1':
            print('MLP prediction layer with BN in the middle')
            self.pred = nn.Sequential(nn.Linear(dim_out, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
        elif pred == 'mlp':
            print('MLP prediction layer without BN')
            self.pred = nn.Sequential(nn.Linear(dim_out, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
        elif pred == 'linbn':
            print('Linear prediction layer with BN')
            self.pred = nn.Sequential(nn.Linear(dim_out, dim_out), BatchNorm1d(dim_out))
        elif pred == 'lin':
            print('Linear prediction layer without BN')
            self.pred = nn.Linear(dim_out, dim_out)
        else:
            self.pred = None

        # initialize the key encoder by the queue encoder
        if method == 'npair':
            self.encoder_k = None
        else:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        # queue for moco
        if (method in ['moco', 'mocon']) and (qlen > 0):
            self.register_buffer("queue", torch.randn(dim, qlen))
            self.queue = nn.functional.normalize(self.queue, dim=0)

            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.queue = self.queue_ptr = None

    def forward(self, im_1, im_2, m=None, criterion=None, imix='none', alpha=1., mix_layers='0', num_aux=0, alpha2=1.):

        bsz = im_1.shape[0]

        # inputmix
        if num_aux > 0:
            im = torch.cat([im_1, im_2], dim=0)
            im = inputmix(im, alpha2, num_aux=num_aux, distributed=self.distributed)
            im_1, im_2 = im[:bsz], im[bsz:]

        # symmetric loss
        if self.sym:
            im_qk = [(im_1, im_2), (im_2, im_1)]
        else:
            im_qk = [(im_1, im_2)]
        glogits = glabels = gloss = None

        for s, (im_q, im_k) in enumerate(im_qk):

            # determine the layer for mix
            if mix_layers == '0':
                mix_layer = 0
            else:
                mix_layer_ind = torch.randint(0, len(mix_layers), ())
                mix_layer = int(mix_layers[mix_layer_ind])

            # i-mix on the input space
            if mix_layer == 0:
                # i-mix
                if imix == 'imixup':
                    im_q, labels_aux, lam = mixup(im_q, alpha)
                elif imix == 'icutmix':
                    im_q, labels_aux, lam = cutmix(im_q, alpha)
                else:
                    labels_aux = lam = None

                # compute query features
                if self.method == 'npair':
                    im_q = torch.cat([im_q, im_k], dim=0)

                q = self.encoder_q(im_q)  # queries: NxC

            # i-mix for npair on the embedding space
            elif self.method == 'npair':
                encoder_q = self.encoder_q.module if hasattr(self.encoder_q, 'module') else self.encoder_q

                im_q = torch.cat([im_q, im_k], dim=0)
                # compute query features before i-mix
                feature_q = encoder_q.forward_partial(im_q, start=0, end=mix_layer)  # queries: NxC
                # i-mix
                if imix == 'imixup':
                    feature_q[:bsz], labels_aux, lam = imixup(feature_q[:bsz], alpha)
                elif imix == 'icutmix':
                    feature_q[:bsz], labels_aux, lam = icutmix(feature_q[:bsz], alpha)
                else:
                    labels_aux = lam = None
                # compute query features after i-mix
                q = encoder_q.forward_partial(feature_q, start=mix_layer, end=100)  # queries: NxC

            # i-mix for moco/byol on the embedding space
            else:
                encoder_q = self.encoder_q.module if hasattr(self.encoder_q, 'module') else self.encoder_q

                # compute query features before i-mix
                feature_q = encoder_q.forward_partial(im_q, start=0, end=mix_layer)  # queries: NxC
                # i-mix
                if imix == 'imixup':
                    feature_q, labels_aux, lam = imixup(feature_q, alpha)
                elif imix == 'icutmix':
                    feature_q, labels_aux, lam = icutmix(feature_q, alpha)
                else:
                    labels_aux = lam = None
                # compute query features after i-mix
                q = encoder_q.forward_partial(feature_q, start=mix_layer, end=100)  # queries: NxC

            # prediction head and normalization
            if self.pred is not None:
                q = self.pred(q)
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            if self.method == 'npair':
                q, k = q[:bsz], q[bsz:]

            else:
                with torch.no_grad():  # no gradient to keys
                    if self.method in ['moco', 'mocon', 'byol']:
                        # update the key encoder
                        if m is None: m = self.emam
                        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                            param_k.data = param_k.data * m + param_q.data * (1. - m)

                    # shuffle BN
                    if self.shuffle_bn:
                        if self.distributed:
                            # make sure no grad in torch.distributed
                            with torch.no_grad():
                                im_k_gather = concat_all_gather(im_k)
                                bsz_all = im_k_gather.shape[0]
                                num_gpus = bsz_all // bsz

                                # random shuffle index
                                ind_shuffle = torch.randperm(bsz_all).cuda()

                                # broadcast to all gpus
                                torch.distributed.broadcast(ind_shuffle, src=0)

                                # index for unshuffle
                                ind_unshuffle = torch.argsort(ind_shuffle)

                                # shuffled index for this gpu
                                gpu = torch.distributed.get_rank()
                                ind_shuffle_this = ind_shuffle.view(num_gpus, -1)[gpu]
                                im_k = im_k_gather[ind_shuffle_this]

                        else:
                            ind_shuffle = torch.randperm(bsz).cuda()
                            im_k = im_k[ind_shuffle]
                            ind_unshuffle = torch.argsort(ind_shuffle)

                    k = self.encoder_k(im_k)  # keys: NxC
                    k = nn.functional.normalize(k, dim=1)

                    # unshuffle BN
                    if self.shuffle_bn:
                        if self.distributed:
                            # make sure no grad in torch.distributed
                            with torch.no_grad():
                                k_gather = concat_all_gather(k)
                                ind_unshuffle_this = ind_unshuffle.view(num_gpus, -1)[gpu]
                                k = k_gather[ind_unshuffle_this]

                        else:
                            k = k[ind_unshuffle]

            # gather keys for revised moco
            if self.distributed and (self.method == 'mocon'):
                k = concat_all_gather(k)

            # compute logits
            if self.method == 'moco':
                # positive logits: Nx1
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                # negative logits: NxK
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)
            elif self.method == 'mocon':
                # contrast: (N+K)xC
                contrast = torch.cat([k, self.queue.clone().detach().t()], dim=0)

                # logits: Nx(N+K)
                logits = torch.mm(q, contrast.t())
            else:
                logits = q.mm(k.t())

            # apply temperature
            if self.method != 'byol':
                logits /= self.temp

            # labels: positive key indicators
            if self.method == 'moco':
                labels = torch.zeros(bsz, dtype=torch.long).cuda()
            elif self.distributed and (self.method == 'mocon'):
                gpu = torch.distributed.get_rank()
                labels = torch.arange(bsz, dtype=torch.long).cuda() + gpu * bsz
                if labels_aux is not None:
                    labels_aux = labels_aux + gpu * bsz
            else:
                labels = torch.arange(bsz, dtype=torch.long).cuda()

            # gather keys for original moco
            if self.distributed and (self.method == 'moco'):
                k = concat_all_gather(k)

            # dequeue and enqueue
            if self.method in ['moco', 'mocon']:
                bsz_all = k.shape[0]
                ptr = self.queue_ptr.item()
                assert self.qlen % bsz_all == 0, 'set qlen % batch_size == 0 for simpliclity'
                self.queue[:, ptr:ptr + bsz_all] = k.T
                ptr = (ptr + bsz_all) % self.qlen
                self.queue_ptr[0] = ptr

            # compute loss, i-mix on the label space
            if self.method == 'byol':
                if imix == 'none':
                    target_logits = logits.diag()
                else:
                    target_logits = lam * logits.diag() + (1. - lam) * logits[range(bsz), labels_aux]
                loss = (2. - 2. * target_logits).mean()
            elif criterion is None:
                loss = None
            elif imix == 'none':
                loss = criterion(logits, labels).mean()
            else:
                loss = (lam * criterion(logits, labels) + (1. - lam) * criterion(logits, labels_aux)).mean()

            if s == 0:
                glogits = logits
                glabels = labels
                gloss = loss
            else:
                glogits = torch.cat([glogits, logits], dim=0)
                glabels = torch.cat([glabels, labels], dim=0)
                gloss = gloss + loss

        return glogits, glabels, gloss

@torch.no_grad()
def concat_all_gather(input):
    gathered = [torch.ones_like(input) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, input, async_op=False)
    return torch.cat(gathered, dim=0)

def inputmix(input, alpha, num_aux=1, pmin=.5, distributed=False):
    if distributed:
        bsz_this = input.shape[0]
        input = concat_all_gather(input)
        bsz = input.shape[0]
        num_gpus = bsz // bsz_this
    else:
        bsz = input.shape[0]
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha] * (num_aux+1)
    if num_aux > 1:
        dist = torch.distributions.dirichlet.Dirichlet(torch.tensor(alpha))
        output = torch.zeros_like(input)
        lam = dist.sample([bsz]).t().to(device=input.device)
        lam = pmin * lam
        lam[0] = lam[0] + pmin
        for i in range(num_aux+1):
            if i == 0:
                randind = torch.arange(bsz, device=input.device)
            else:
                randind = torch.randperm(bsz, device=input.device)
            lam_expanded = lam[i].view([-1] + [1]*(input.dim()-1))
            output += lam_expanded * input[randind]
    else:
        beta = torch.distributions.beta.Beta(*alpha)
        randind = torch.randperm(bsz, device=input.device)
        lam = beta.sample([bsz]).to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
        output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    if distributed:
        gpu = torch.distributed.get_rank()
        return output[gpu*bsz_this:(gpu+1)*bsz_this]
    else:
        return output

def mixup(input, alpha, share_lam=False):
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    if share_lam:
        lam = beta.sample().to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam
    else:
        lam = beta.sample([input.shape[0]]).to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
    output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return output, randind, lam

def cutmix(input, alpha):
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)
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

class BatchNorm1d(nn.Module):
    def __init__(self, dim, affine=True, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine, momentum=momentum)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

