import torch
import torch.nn as nn

CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')


def chunk_feature(feature, chunk):

    if chunk == 1:
        return feature
    # B x C x H x W => (B*chunk^2) x C x (H//chunk) x (W//chunk)
    _f_new = torch.chunk(feature, chunk, dim=2)
    _f_new = [torch.chunk(f, chunk, dim=3) for f in _f_new]
    f_new = []
    for f in _f_new:
        f_new += f
    f_new = torch.cat(f_new, dim=0)
    return f_new

class Builder(nn.Module):

    def __init__(self, base_encoder, dim=128,  K=65536, m=0.999, T=0.07, qlen=4096, emam=0.999, temp=0.2,  mlp=True, proj='lin', pred='none', method='npair', shuffle_bn=False, head_mul=1, sym=False, in_channels=3,
        small=False, distributed=False, kaiming_init=True, stages=[4], task='old', base_encoder_kwargs={}, get_head=None, chunks=[1], approach='B'):
        super(Builder, self).__init__()

        self.qlen = qlen
        self.stages = stages
        self.emam = emam
        self.temp = temp
        self.method = method
        self.shuffle_bn = shuffle_bn
        self.sym = sym
        self.distributed = distributed
        self.task = task
        self.mlp = mlp
        self.m = m
        self.chunks = chunks  # chunk feature (segmentation)
        self.T = T
        self.K = K
        self.approach = approach

        # encoder
        #self.encoder_q = base_encoder(num_classes=dim*head_mul, in_channels=in_channels, small=small, kaiming_init=kaiming_init)
        #self.encoder_k = base_encoder(num_classes=dim*head_mul, in_channels=in_channels, small=small, kaiming_init=kaiming_init)
        self.encoder_q = base_encoder(num_classes=dim, pretrained=True, kaiming_init=kaiming_init,**base_encoder_kwargs)  # ######
        self.encoder_k = base_encoder(num_classes=dim, pretrained=True, kaiming_init=kaiming_init,**base_encoder_kwargs)  # ######


        """
        checkpoint = torch.load('./checkpoints/checkpoint_0800.pth.tar', map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        msg = self.encoder_k.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
        msg = self.encoder_q.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        """
        if mlp:
            fc_q = {}
            fc_k = {}
            for stage in stages:
                if stage > 0:
                    try:
                        # BottleNeck
                        dim_mlp = getattr(self.encoder_q, "layer%d" %
                                          stage)[-1].conv2.weight.size()[0]
                    except torch.nn.modules.module.ModuleAttributeError:
                        # BasicBlock
                        dim_mlp = getattr(self.encoder_q, "layer%d" %
                                          stage)[-1].conv2.weight.size()[0]
                elif stage == 0:
                    dim_mlp = self.encoder_q.conv1.weight.size()[0]
                fc_q["stage%d" % (stage)] = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
                fc_k["stage%d" % (stage)] = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_q.fc_csg = nn.ModuleDict(fc_q)
            self.encoder_k.fc_csg = nn.ModuleDict(fc_k)
            for param_q, param_k in zip(self.encoder_q.fc_csg.parameters(), self.encoder_k.fc_csg.parameters()):
                param_k.data.copy_(param_q.data)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        if type(self.encoder_q).__name__ == "ResNet":
            try:
                # BottleNeck
                dims = [self.encoder_q.conv1.weight.size()[0]] + [getattr(self.encoder_q, "layer%d"%stage)[-1].conv3.weight.size()[0] for stage in range(1, 5)]
            except:
                # BasicBlock
                dims = [self.encoder_q.conv1.weight.size()[0]] + [getattr(self.encoder_q, "layer%d"%stage)[-1].conv2.weight.size()[0] for stage in range(1, 5)]
        elif type(self.encoder_q).__name__ == "DigitNet":
                dims = [64 for stage in range(1, 5)]
        for stage in stages:
            self.register_buffer("queue%d"%(stage), torch.randn(dim, K))
            setattr(self, "queue%d"%(stage), nn.functional.normalize(getattr(self, "queue%d"%(stage)), dim=0))
            self.register_buffer("queue_ptr%d"%(stage), torch.zeros(1, dtype=torch.long))

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
    
        # queue for moco
        if (method in ['moco', 'mocon']) and (qlen > 0):
            self.register_buffer("queue", torch.randn(dim, qlen))
            self.queue = nn.functional.normalize(self.queue, dim=0)

            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.queue = self.queue_ptr = None

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.fc_csg.parameters(), self.encoder_k.fc_csg.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, stage):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(getattr(self, "queue_ptr%d"%(stage)))

        if ptr + batch_size <= self.K:
            getattr(self, "queue%d"%(stage))[:, ptr:ptr + batch_size] = keys.T
        else:
            getattr(self, "queue%d"%(stage))[:, ptr:] = keys[:(self.K - ptr)].T
            getattr(self, "queue%d"%(stage))[:, :ptr + batch_size - self.K] = keys[:(ptr + batch_size - self.K)].T
        ptr = (ptr + batch_size) % self.K  # move pointer
        getattr(self, "queue_ptr%d"%(stage))[0] = ptr

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
                    im_q_mix, labels_aux, lam = mixup(im_q, alpha)
                elif imix == 'icutmix':
                    im_q_mix_0, labels_aux_0, lam_0 = cutmix(im_q, alpha)
                    im_q_mix_1, labels_aux_1, lam_1 = cutmix(im_q, alpha)
                    im_q_mix_2, labels_aux_2, lam_2 = cutmix(im_q, alpha)
                    im_q_mix_3, labels_aux_3, lam_3 = cutmix(im_q, alpha)
                    im_q_mix_4, labels_aux_4, lam_4 = cutmix(im_q, alpha)
                    im_q_mix_5, labels_aux_5, lam_5 = cutmix(im_q, alpha)
                else:
                    labels_aux = lam = None

                # compute query features
                if self.method == 'npair':
                    if self.approach == 'A':
                        im_q = torch.cat([im_q_mix, im_k], dim=0)
                    else:
                        im_q = torch.cat([im_q, im_q_mix_0, im_q_mix_1, im_q_mix_2, im_q_mix_3, im_q_mix_4, im_q_mix_5, im_k], dim=0)
                        #im_q = torch.cat([im_q, im_q_mix_0, im_q_mix_1, im_k], dim=0)

                q, features_new = self.encoder_q(im_q, output_features=["layer%d"%stage for stage in self.stages], task=self.task)  # queries: NxC
            # i-mix for npair on the embedding space

            # prediction head and normalization

            q = nn.functional.normalize(q, dim=1)

            # compute key features
            if self.method == 'npair':
                if self.approach == 'A':
                    q, k = q[:bsz], q[bsz:]
                else: 
                    q, q_0, q_1, q_2, q_3, q_4, q_5, k = q[:bsz], q[bsz:int(2*bsz)], q[int(2*bsz):int(3*bsz)], q[int(3*bsz):int(4*bsz)], q[int(4*bsz):int(5*bsz)], q[int(5*bsz):int(6*bsz)], q[int(6*bsz):int(7*bsz)], q[int(7*bsz):] 
                    #q, q_1, k = q[bsz:int(2*bsz)], q[int(2*bsz):int(3*bsz)], q[int(3*bsz):]

            else:
                with torch.no_grad():  # no gradient to keys
                    if self.method in ['moco', 'mocon', 'byol']:
                        # update the key encoder
                        if m is None: m = self.emam
                        for param_q, param_k in zip(self.encoder_q.fc_csg.parameters(), self.encoder_k.fc_csg.parameters()):
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

                    #k, _ = self.encoder_k(im_k)  # keys: NxC

                    k, features_k = self.encoder_q(im_k, output_features=["layer%d"%stage for stage in self.stages], task=self.task)  # queries: NxC
                    #q, features_q = self.encoder_q(im_q, output_features=["layer%d"%stage for stage in self.stages], task=self.task)  # queries: NxC

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
                #logits = q.mm(k.t())
                logits_0 = q_0.mm(k.t())
                logits_1 = q_1.mm(k.t())                               
                logits_2 = q_2.mm(k.t())
                logits_3 = q_3.mm(k.t())      
                logits_4 = q_4.mm(k.t())  
                logits_5 = q_5.mm(k.t())           
            # apply temperature
            if self.method != 'byol':
                #logits /= self.temp
                logits_0 /= self.temp
                logits_1 /= self.temp
                logits_2 /= self.temp
                logits_3 /= self.temp
                logits_4 /= self.temp
                logits_5 /= self.temp               

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
                #loss = (lam_0 * criterion(logits, labels) + (1. - lam_0) * criterion(logits, labels_aux_0)).mean()
                with torch.no_grad():

                    loss_0 = (lam_0 * criterion(logits_0, labels) + (1. - lam_0) * criterion(logits_0, labels_aux_0)).mean()
                    loss_1 = (lam_1 * criterion(logits_1, labels) + (1. - lam_1) * criterion(logits_1, labels_aux_1)).mean()
                    loss_2 = (lam_2 * criterion(logits_2, labels) + (1. - lam_2) * criterion(logits_2, labels_aux_2)).mean()
                    loss_3 = (lam_3 * criterion(logits_3, labels) + (1. - lam_3) * criterion(logits_3, labels_aux_3)).mean()
                    loss_4 = (lam_4 * criterion(logits_4, labels) + (1. - lam_4) * criterion(logits_4, labels_aux_4)).mean()
                    loss_5 = (lam_5 * criterion(logits_5, labels) + (1. - lam_5) * criterion(logits_5, labels_aux_5)).mean()


                    loss = max(loss_0, loss_1, loss_2, loss_3, loss_4, loss_5)

            if s == 0:
                if loss == loss_0:
                    glogits = logits_0
                    loss = (lam_0 * criterion(logits_0, labels) + (1. - lam_0) * criterion(logits_0, labels_aux_0)).mean()
                elif loss == loss_1:
                    glogits = logits_1
                    loss = (lam_1 * criterion(logits_1, labels) + (1. - lam_1) * criterion(logits_1, labels_aux_1)).mean()
                elif loss == loss_2:
                    glogits = logits_2
                    loss = (lam_2 * criterion(logits_2, labels) + (1. - lam_2) * criterion(logits_2, labels_aux_2)).mean()
                elif loss == loss_3:
                    glogits = logits_3
                    loss = (lam_3 * criterion(logits_3, labels) + (1. - lam_3) * criterion(logits_3, labels_aux_3)).mean()
                elif loss == loss_4:
                    glogits = logits_4
                    loss = (lam_4 * criterion(logits_4, labels) + (1. - lam_4) * criterion(logits_4, labels_aux_4)).mean()
                elif loss == loss_5:
                    glogits = logits_5
                    loss = (lam_5 * criterion(logits_5, labels) + (1. - lam_5) * criterion(logits_5, labels_aux_5)).mean()
                #glogits = logits
                glabels = labels
                gloss = loss
            else:
                glogits = torch.cat([glogits, logits], dim=0)
                glabels = torch.cat([glabels, labels], dim=0)
                gloss = gloss + loss
        results = {}

        results['predictions_csg'] = []
        results['targets_csg'] = []
        with torch.no_grad():  # no gradient to keys
            if self.mlp:
                self._momentum_update_key_encoder()  # update the key encoder
        _, features_old = self.encoder_k.forward_backbone(im_k, output_features=["layer%d"%stage for stage in self.stages])

        for idx, stage in enumerate(self.stages):
            chunk = self.chunks[idx]
            # compute query features

            features_ = chunk_feature(features_new["layer%d"%stage], chunk)
            q_feature = features_[:bsz]

            if self.mlp:
                q_ = self.encoder_q.fc_csg["stage%d"%(stage)](self.encoder_q.avgpool(q_feature).view(q_feature.shape[0], -1))

            q_ = nn.functional.normalize(q_, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                k_feature = chunk_feature(features_old["layer%d"%stage], chunk)
                # A-Pool #############
                if self.mlp:
                    k_ = self.encoder_k.fc_csg["stage%d"%(stage)](self.encoder_k.avgpool(k_feature).view(features_old["layer%d"%stage].size(0)*chunk**2, -1))
                # #####################
                k_ = nn.functional.normalize(k_, dim=1)

            # compute logits
            # positive logits: Nx1

            l_pos = torch.einsum('nc,nc->n', [q_, k_]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q_, getattr(self, "queue%d"%(stage)).clone().detach()])
            # logits: Nx(1+K)
            logits_ = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits_ /= self.T
            # labels: positive key indicators
            labels_ = torch.zeros(logits_.shape[0], dtype=torch.long).cuda()
            self._dequeue_and_enqueue(k, stage)
            results['predictions_csg'].append(logits_)
            results['targets_csg'].append(labels_)


            for idx in range(len(self.stages)):
                # predictions: cosine b/w q and k
                # targets: zeros
                csg_loss = CrossEntropyLoss(results['predictions_csg'][idx], results['targets_csg'][idx])

        return glogits, glabels, gloss, csg_loss, results

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

