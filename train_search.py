import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype
from architect import Architect

# python train_search.py \\
#        --tmp_data_dir /path/to/your/data \\
#        --save log_path \\
#        --add_layers 6 \\
#        --add_layers 12 \\
#        --dropout_rate 0.1 \\
#        --dropout_rate 0.4 \\
#        --dropout_rate 0.7 \\
#        --note note_of_this_run

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='TMP', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='tmp', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=['0.1', '0.4', '0.7'], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0', '6', '12'], help='add layers')
parser.add_argument('--c_lambda', type=float, default=0.05, help='cooperative learning coefficient')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')

args = parser.parse_args()

args.save = '/ceph/aseem-volume/full/search/05_12/logging'
args.tmp_data_dir = '/ceph/aseem-volume/full/search/05_12/data'
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'

c_lambda = args.c_lambda

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    #  prepare dataset
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)

    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal_1 = copy.deepcopy(switches)
    switches_reduce_1 = copy.deepcopy(switches)

    switches_normal_2 = copy.deepcopy(switches)
    switches_reduce_2 = copy.deepcopy(switches)

    # To be moved to args
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 6, 12]
    if len(args.dropout_rate) ==3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0]
    eps_no_archs = [10, 10, 10]
    for sp in range(len(num_to_keep)):

        model_1 = Network(args.init_channels + int(add_width[sp]), CIFAR_CLASSES, args.layers + int(add_layers[sp]), criterion, switches_normal=switches_normal_1, switches_reduce=switches_reduce_1, p=float(drop_rate[sp]))
        model_1 = nn.DataParallel(model_1)
        model_1 = model_1.cuda()

        model_2 = Network(args.init_channels + int(add_width[sp]), CIFAR_CLASSES, args.layers + int(add_layers[sp]), criterion, switches_normal=switches_normal_2, switches_reduce=switches_reduce_2, p=float(drop_rate[sp]))
        model_2 = nn.DataParallel(model_2)
        model_2 = model_2.cuda()

        logging.info("param size 1= %fMB", utils.count_parameters_in_MB(model_1))
        logging.info("param size 2= %fMB", utils.count_parameters_in_MB(model_2))

        network_params_1 = []
        for k, v in model_1.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params_1.append(v)

        network_params_2 = []
        for k, v in model_2.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params_2.append(v)

        optimizer_1 = torch.optim.SGD(
                network_params_1,
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)

        optimizer_2 = torch.optim.SGD(
                network_params_2,
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)

        optimizer_a_1 = torch.optim.Adam(model_1.module.arch_parameters(),
                    lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

        optimizer_a_2 = torch.optim.Adam(model_2.module.arch_parameters(),
                    lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

        scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_1, float(args.epochs), eta_min=args.learning_rate_min)

        scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_2, float(args.epochs), eta_min=args.learning_rate_min)

        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2

        architect = Architect(model_1, model_2, network_params_1, network_params_2, criterion, args)

        for epoch in range(epochs):
            scheduler_1.step()
            scheduler_2.step()

            lr_1 = scheduler_1.get_lr()[0]
            lr_2 = scheduler_2.get_lr()[0]

            logging.info('Epoch: %d lr_1: %e lr_2: %e', epoch, lr_1, lr_2)
            epoch_start = time.time()

            # training
            if epoch < eps_no_arch:

                model_1.module.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model_1.module.update_p()

                model_2.module.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model_2.module.update_p()

                train_acc_1, train_acc_2, train_obj = train(architect, train_queue, valid_queue, model_1, model_2, network_params_1, network_params_2, criterion, optimizer_1, optimizer_2, optimizer_a_1, optimizer_a_2, lr_1, lr_2, train_arch=False)
            else:

                model_1.module.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                model_1.module.update_p()

                model_2.module.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor)
                model_2.module.update_p()

                train_acc_1, train_acc_2, train_obj = train(architect, train_queue, valid_queue, model_1, model_2, network_params_1, network_params_2, criterion, optimizer_1, optimizer_2, optimizer_a_1, optimizer_a_2, lr_1, lr_2, train_arch=True)

            logging.info('Train_acc %f %f', train_acc_1, train_acc_2)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            # validation
            if epochs - epoch < 5:
                valid_acc_1, valid_obj_1, valid_acc_2, valid_obj_2 = infer(valid_queue, model_1, model_2, criterion)
                logging.info('Valid_acc %f %f', valid_acc_1, valid_acc_2)
        utils.save(model_1, os.path.join(args.save, 'weights_1.pt'))
        utils.save(model_2, os.path.join(args.save, 'weights_2.pt'))
        print('------Dropping %d paths------' % num_to_drop[sp])
        # Save switches info for s-c refinement.
        if sp == len(num_to_keep) - 1:
            switches_normal_1_copy = copy.deepcopy(switches_normal_1)
            switches_reduce_1_copy = copy.deepcopy(switches_reduce_1)

            switches_normal_2_copy = copy.deepcopy(switches_normal_2)
            switches_reduce_2_copy = copy.deepcopy(switches_reduce_2)

        # drop operations with low architecture weights
        arch_param_1 = model_1.module.arch_parameters()
        normal_prob_1 = F.softmax(arch_param_1[0], dim=sm_dim).data.cpu().numpy()

        arch_param_2 = model_2.module.arch_parameters()
        normal_prob_2 = F.softmax(arch_param_2[0], dim=sm_dim).data.cpu().numpy()

        for i in range(14):
            idxs_1 = []
            idxs_2 = []
            for j in range(len(PRIMITIVES)):
                if switches_normal_1[i][j]:
                    idxs_1.append(j)
                if switches_normal_2[i][j]:
                    idxs_2.append(j)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop_1 = get_min_k_no_zero(normal_prob_1[i, :], idxs_1, num_to_drop[sp])
                drop_2 = get_min_k_no_zero(normal_prob_2[i, :], idxs_2, num_to_drop[sp])
            else:
                drop_1 = get_min_k(normal_prob_1[i, :], num_to_drop[sp])
                drop_2 = get_min_k(normal_prob_2[i, :], num_to_drop[sp])
            for idx in drop_1:
                switches_normal_1[i][idxs_1[idx]] = False
            for idx in drop_2:
                switches_normal_2[i][idxs_2[idx]] = False

        reduce_prob_1 = F.softmax(arch_param_1[1], dim=-1).data.cpu().numpy()
        reduce_prob_2 = F.softmax(arch_param_2[1], dim=-1).data.cpu().numpy()

        for i in range(14):
            idxs_1 = []
            idxs_2 = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce_1[i][j]:
                    idxs_1.append(j)
                if switches_reduce_2[i][j]:
                    idxs_2.append(j)
            if sp == len(num_to_keep) - 1:
                drop_1 = get_min_k_no_zero(reduce_prob_1[i, :], idxs_1, num_to_drop[sp])
                drop_2 = get_min_k_no_zero(reduce_prob_2[i, :], idxs_2, num_to_drop[sp])
            else:
                drop_1 = get_min_k(reduce_prob_1[i, :], num_to_drop[sp])
                drop_2 = get_min_k(reduce_prob_2[i, :], num_to_drop[sp])
            for idx in drop_1:
                switches_reduce_1[i][idxs_1[idx]] = False
            for idx in drop_2:
                switches_reduce_2[i][idxs_2[idx]] = False

        logging.info('switches_normal_1 = %s', switches_normal_1)
        logging_switches(switches_normal_1)
        logging.info('switches_reduce_1 = %s', switches_reduce_1)
        logging_switches(switches_reduce_1)

        logging.info('switches_normal_2 = %s', switches_normal_2)
        logging_switches(switches_normal_2)
        logging.info('switches_reduce_2 = %s', switches_reduce_2)
        logging_switches(switches_reduce_2)

        if sp == len(num_to_keep) - 1:
            arch_param_1 = model_1.module.arch_parameters()
            arch_param_2 = model_2.module.arch_parameters()

            normal_prob_1 = F.softmax(arch_param_1[0], dim=sm_dim).data.cpu().numpy()
            normal_prob_2 = F.softmax(arch_param_2[0], dim=sm_dim).data.cpu().numpy()

            reduce_prob_1 = F.softmax(arch_param_1[1], dim=sm_dim).data.cpu().numpy()
            reduce_prob_2 = F.softmax(arch_param_2[1], dim=sm_dim).data.cpu().numpy()

            normal_final_1 = [0 for idx in range(14)]
            reduce_final_1 = [0 for idx in range(14)]

            normal_final_2 = [0 for idx in range(14)]
            reduce_final_2 = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_1_copy[i][0] == True:
                    normal_prob_1[i][0] = 0
                normal_final_1[i] = max(normal_prob_1[i])

                if switches_normal_2_copy[i][0] == True:
                    normal_prob_2[i][0] = 0
                normal_final_2[i] = max(normal_prob_2[i])

                if switches_reduce_1_copy[i][0] == True:
                    reduce_prob_1[i][0] = 0
                reduce_final_1[i] = max(reduce_prob_1[i])

                if switches_reduce_2_copy[i][0] == True:
                    reduce_prob_2[i][0] = 0
                reduce_final_2[i] = max(reduce_prob_2[i])

            # Generate Architecture, similar to DARTS
            keep_normal_1 = [0, 1]
            keep_reduce_1 = [0, 1]

            keep_normal_2 = [0, 1]
            keep_reduce_2 = [0, 1]

            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn_1 = normal_final_1[start:end]
                tbsr_1 = reduce_final_1[start:end]

                tbsn_2 = normal_final_2[start:end]
                tbsr_2 = reduce_final_2[start:end]

                edge_n_1 = sorted(range(n), key=lambda x: tbsn_1[x])
                edge_n_2 = sorted(range(n), key=lambda x: tbsn_2[x])

                keep_normal_1.append(edge_n_1[-1] + start)
                keep_normal_1.append(edge_n_1[-2] + start)

                keep_normal_2.append(edge_n_2[-1] + start)
                keep_normal_2.append(edge_n_2[-2] + start)

                edge_r_1 = sorted(range(n), key=lambda x: tbsr_1[x])
                edge_r_2 = sorted(range(n), key=lambda x: tbsr_2[x])

                keep_reduce_1.append(edge_r_1[-1] + start)
                keep_reduce_1.append(edge_r_1[-2] + start)

                keep_reduce_2.append(edge_r_2[-1] + start)
                keep_reduce_2.append(edge_r_2[-2] + start)

                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):

                if not i in keep_normal_1:
                    for j in range(len(PRIMITIVES)):
                        switches_normal_1[i][j] = False
                if not i in keep_reduce_1:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce_1[i][j] = False

                if not i in keep_normal_2:
                    for j in range(len(PRIMITIVES)):
                        switches_normal_2[i][j] = False
                if not i in keep_reduce_2:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce_2[i][j] = False

            # translate switches into genotype
            genotype_1 = parse_network(switches_normal_1, switches_reduce_1)
            logging.info(genotype_1)
            genotype_2 = parse_network(switches_normal_2, switches_reduce_2)
            logging.info(genotype_2)

            ## restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')

            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks
                num_sk_1 = check_sk_number(switches_normal_1)
                if not num_sk_1 > max_sk:
                    continue
                while num_sk_1 > max_sk:
                    normal_prob_1 = delete_min_sk_prob(switches_normal_1, switches_normal_1_copy, normal_prob_1)
                    switches_normal_1 = keep_1_on(switches_normal_1_copy, normal_prob_1)
                    switches_normal_1 = keep_2_branches(switches_normal_1, normal_prob_1)
                    num_sk_1 = check_sk_number(switches_normal_1)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype_1 = parse_network(switches_normal_1, switches_reduce_1)
                logging.info(genotype_1)

            for sks in range(0, 9):
                max_sk = 8 - sks
                num_sk_2 = check_sk_number(switches_normal_2)
                if not num_sk_2 > max_sk:
                    continue
                while num_sk_2 > max_sk:
                    normal_prob_2 = delete_min_sk_prob(switches_normal_2, switches_normal_2_copy, normal_prob_2)
                    switches_normal_2 = keep_1_on(switches_normal_2_copy, normal_prob_2)
                    switches_normal_2 = keep_2_branches(switches_normal_2, normal_prob_2)
                    num_sk_2 = check_sk_number(switches_normal_2)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype_2 = parse_network(switches_normal_2, switches_reduce_2)
                logging.info(genotype_2)

def train(architect, train_queue, valid_queue, model_1, model_2, network_params_1, network_params_2, criterion, optimizer_1, optimizer_2, optimizer_a_1, optimizer_a_2, lr_1, lr_2, train_arch=True):
    objs = utils.AvgrageMeter()
    top1_1 = utils.AvgrageMeter()
    top5_1 = utils.AvgrageMeter()

    top1_2 = utils.AvgrageMeter()
    top5_2 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model_1.train()
        model_2.train()

        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above.
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)

            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

            optimizer_a_1.zero_grad()
            optimizer_a_2.zero_grad()
            
            logits_1 = model_1(input_search)
            logits_2 = model_2(input_search)

            loss_1 = criterion(logits_1, target_search)
            loss_2 = criterion(logits_2, target_search)

            loss = loss_1 + loss_2
            loss.backward()
            # architect.step(input, target, input_search, target_search, lr_1, optimizer_1, optimizer_2, optimizer_a_1, optimizer_a_2)
            
            nn.utils.clip_grad_norm_(model_1.module.arch_parameters(), args.grad_clip)
            nn.utils.clip_grad_norm_(model_2.module.arch_parameters(), args.grad_clip)

            optimizer_a_1.step()
            optimizer_a_2.step()

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        # loss = architect.compute_loss(input, target)
        logits_1 = model_1(input)
        logits_2 = model_2(input)

        loss_1 = criterion(logits_1, target)
        loss_2 = criterion(logits_2, target)

        loss = loss_1 + loss_2
        loss.backward()

        nn.utils.clip_grad_norm_(network_params_1, args.grad_clip)
        nn.utils.clip_grad_norm_(network_params_2, args.grad_clip)
        optimizer_1.step()
        optimizer_2.step()

        # logits_1 = model_1(input)
        # logits_2 = model_2(input)

        prec1_1, prec5_1 = utils.accuracy(logits_1, target, topk=(1, 5))
        prec1_2, prec5_2 = utils.accuracy(logits_2, target, topk=(1, 5))

        objs.update(loss.data.item(), n)

        top1_1.update(prec1_1.data.item(), n)
        top5_1.update(prec5_1.data.item(), n)

        top1_2.update(prec1_2.data.item(), n)
        top5_2.update(prec5_2.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f R1: %f R5: %f', step, objs.avg, top1_1.avg, top5_1.avg, top1_2.avg, top5_2.avg)

    return top1_1.avg, top1_2.avg, objs.avg

def infer(valid_queue, model_1, model_2, criterion):
    objs_1 = utils.AvgrageMeter()
    top1_1 = utils.AvgrageMeter()
    top5_1 = utils.AvgrageMeter()

    objs_2 = utils.AvgrageMeter()
    top1_2 = utils.AvgrageMeter()
    top5_2 = utils.AvgrageMeter()

    model_1.eval()
    model_2.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits_1 = model_1(input)
            loss_1 = criterion(logits_1, target)
            logits_2 = model_2(input)
            loss_2 = criterion(logits_2, target)

        prec1_1, prec5_1 = utils.accuracy(logits_1, target, topk=(1, 5))
        prec1_2, prec5_2 = utils.accuracy(logits_2, target, topk=(1, 5))

        n = input.size(0)

        objs_1.update(loss_1.data.item(), n)
        top1_1.update(prec1_1.data.item(), n)
        top5_1.update(prec5_1.data.item(), n)

        objs_2.update(loss_2.data.item(), n)
        top1_2.update(prec1_2.data.item(), n)
        top5_2.update(prec5_2.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f %e %f %f', step, objs_1.avg, top1_1.avg, top5_1.avg, objs_2.avg, top1_2.avg, top5_2.avg)

    return top1_1.avg, objs_1.avg, top1_2.avg, objs_2.avg


def parse_network(switches_normal, switches_reduce):

    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)

    concat = range(2, 6)

    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )

    return genotype

def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1

    return index
def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index

def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)

def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1

    return count

def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0

    return probs_out

def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False
    return switches

def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False
    return switches

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
