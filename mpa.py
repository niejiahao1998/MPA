from model.MPA_matching import MPA_MatchingNet
from util.utils import count_params, set_seed, mIOU

import argparse
from copy import deepcopy
import os
import time
import torch
from torch.nn import CrossEntropyLoss, DataParallel
import torch.nn.functional as F
from torch.optim import SGD
from tqdm import tqdm
from data.dataset import FSSDataset


def parse_args():
    parser = argparse.ArgumentParser(description='MPA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        required=True,
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='fss',
                        choices=['fss', 'deepglobe', 'isic', 'lung','suim'],
                        help='training dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help='batch size of training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--crop-size',
                        type=int,
                        default=473,
                        help='cropping size of training samples')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--refine', dest='refine', action='store_true', default=False)
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--episode',
                        type=int,
                        default=48000,
                        help='total episodes of training')
    parser.add_argument('--snapshot',
                        type=int,
                        default=1200,
                        help='save the model after each snapshot episodes')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate testing samples')

    
    parser.add_argument('--base-q', type=float, default=1, help="The parameter for Base Query Loss")
    parser.add_argument('--base-s', type=float, default=0.2, help="The parameter for Base Support Loss")
    parser.add_argument('--parallel-q', type=float, default=1, help="The parameter for Query Loss on parallel path")
    parser.add_argument('--parallel-s', type=float, default=0.4, help="The parameter for Support Loss on parallel path")
    parser.add_argument('--chain', type=float, default=0.1, help="The parameter for chain path")

    args = parser.parse_args()
    return args

def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    if args.dataset == 'fss':
        num_classes = 1000
    elif args.dataset == 'deepglobe':
        num_classes = 6
    elif args.dataset == 'isic':
        num_classes = 3
    elif args.dataset == 'lung':
        num_classes = 1
    elif args.dataset == 'suim':
        num_classes = 7

    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):

        img_s_list = img_s_list.permute(1,0,2,3,4)
        mask_s_list = mask_s_list.permute(1,0,2,3)
            
        img_s_list = img_s_list.numpy().tolist()
        mask_s_list = mask_s_list.numpy().tolist()

        img_q, mask_q = img_q.cuda(), mask_q.cuda()

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
            img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()
        cls = cls[0].item()
        cls = cls + 1

        with torch.no_grad():
            out_ls = model(img_s_list, mask_s_list, [img_q], [mask_q])
            pred = torch.argmax(out_ls[0], dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def adjust_batch_size(bsz, dataset):
    NUM_CLASSES = {
    'deepglobe': 6,
    'isic': 3,
    'suim': 7,
    }

    if dataset not in NUM_CLASSES:
        return bsz

    num_classes = NUM_CLASSES[dataset]

    if bsz < num_classes:
        return bsz
    
    return (bsz // num_classes) * num_classes

def loss_calculator(parallel_out_ls, chain_out_ls, self_out_ls, mask_q_list, mask_s, refine, criterion):
    if refine:
        # Parallel Query and Support
        p_query_loss = criterion(parallel_out_ls[0],mask_q_list[0])
        p_support_loss = 0
        for idx in range(len(mask_q_list)):
            p_query_loss += criterion(parallel_out_ls[idx+1][0],mask_q_list[idx])
            p_support_loss += criterion(parallel_out_ls[idx+1][1],mask_s)
    
    else:
        # Parallel Query and Support
        p_query_loss = 0
        p_support_loss = 0
        for idx in range(len(mask_q_list)):
            p_query_loss += criterion(parallel_out_ls[idx][0],mask_q_list[idx])
            p_support_loss += criterion(parallel_out_ls[idx][1],mask_s)

    # Chain
    c_loss = 0
    for idx in range(len(mask_q_list)-1):
        c_support_loss = criterion(chain_out_ls[idx][1],mask_s)
        c_query_loss = criterion(chain_out_ls[idx][0],mask_q_list[idx+1])
        c_loss = c_loss + c_support_loss + c_query_loss
    
    # Base query
    self_query_loss = 0
    for idx in range(len(mask_q_list)):
        self_query_loss += criterion(self_out_ls[idx],mask_q_list[idx])

    # Base support
    self_support_loss = criterion(self_out_ls[-1],mask_s)

    return p_query_loss, p_support_loss, c_loss, self_query_loss, self_support_loss

def main():
    path_dir = 'mpa'

    args = parse_args()
    print('\n' + str(args))
    
    miou = 0
    save_path = 'outdir/models/%s/%s' % (args.dataset, path_dir)
    os.makedirs(save_path, exist_ok=True)
 
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    train_dataset = args.dataset+'mpa'
    batch_size = adjust_batch_size(args.batch_size, args.dataset)
    trainloader = FSSDataset.build_dataloader(train_dataset, batch_size, 4, '0', 'val', args.shot)
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, batch_size, 4, '0', 'val', args.shot)

    print('Do we use SSP refinement?', args.refine)
    model = MPA_MatchingNet(args.backbone, args.refine, args.shot)
    print('\nParams: %.1fM' % count_params(model))

    # Print experiment settings
    # print('\nExperiment settings: ')
    # print(f"    Base Query Parameter: {args.base_q}")
    # print(f"    Base Support Parameter: {args.base_s}")
    # print(f"    Parallel Path Query Parameter: {args.parallel_q}")
    # print(f"    Parallel Path Support Parameter: {args.parallel_s}")
    # print(f"    Chain Path Parameter: {args.chain}")

    base_query = args.base_q
    base_support = args.base_s
    parallel_query = args.parallel_q
    parallel_support = args.parallel_s
    chain = args.chain

    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = SGD([param for param in model.parameters() if param.requires_grad],
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)

    model = DataParallel(model).cuda()
    best_model = None

    iters = 0
    total_iters = args.episode // batch_size
    lr_decay_iters = [total_iters // 3, total_iters * 2 // 3]

    previous_best = float(miou)
    
    phase_max = 6
    phase = 1
    previous_phase_miou = 0
    counter = 0
    for epoch in range(args.episode // args.snapshot):
        
        print('Phase:', f'{phase} queries loaded')
        print("\n==> Epoch %i, learning rate = %.5f\t\t\t\t Previous best = %.2f"
              % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0

        tbar = tqdm(trainloader)
        set_seed(int(time.time()))

        for i, (img_s_list, mask_s_list, img_q_list, mask_q_list, _, _, _) in enumerate(tbar):

            img_s_list = img_s_list.permute(1,0,2,3,4)
            mask_s_list = mask_s_list.permute(1,0,2,3)      
            img_s_list = img_s_list.numpy().tolist()
            mask_s_list = mask_s_list.numpy().tolist()

            img_q_list = img_q_list.permute(1,0,2,3,4)
            mask_q_list = mask_q_list.permute(1,0,2,3)      
            img_q_list = img_q_list.numpy().tolist()
            mask_q_list = mask_q_list.numpy().tolist()

            img_q_list = img_q_list[:phase]
            mask_q_list = mask_q_list[:phase]

            for k in range(len(img_s_list)):
                img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
                img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()

            # load queries
            
            assert len(img_q_list) == phase, 'Wrong query image number!'
            assert len(mask_q_list) == phase, 'Wrong query mask number!'

            for k in range(len(img_q_list)):
                img_q_list[k], mask_q_list[k] = torch.Tensor(img_q_list[k]), torch.Tensor(mask_q_list[k])
                img_q_list[k], mask_q_list[k] = img_q_list[k].cuda(), mask_q_list[k].cuda()

            parallel_out_ls, chain_out_ls, self_out_ls = model(img_s_list, mask_s_list, img_q_list, mask_q_list)
            
            mask_s = torch.cat(mask_s_list, dim=0)
            mask_s = mask_s.long()

            mask_q_list = [mask.long() for mask in mask_q_list]

            p_query_loss, p_support_loss, c_loss, self_query_loss, self_support_loss = loss_calculator(parallel_out_ls, chain_out_ls, self_out_ls, mask_q_list, mask_s, args.refine, criterion)

            loss = self_support_loss * base_support + self_query_loss * base_query + p_support_loss * parallel_support + p_query_loss * parallel_query + c_loss * chain

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            if iters in lr_decay_iters:
                optimizer.param_groups[0]['lr'] /= 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        model.eval()
        set_seed(args.seed)
        miou = evaluate(model, testloader, args)


        if miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = miou
            torch.save(best_model.module.state_dict(),
                os.path.join(save_path, '%s_%ishot_%.2f.pth' % (args.backbone, args.shot, miou)))
        
        
        if miou >= previous_best*1.01:
            counter = 0
        else:
            counter += 1

        if counter >= 3 and phase < phase_max:
            phase += 1
            counter = 0
            print(f"Phase updated to {phase}.")
            
    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

    torch.save(best_model.module.state_dict(),
               os.path.join(save_path, '%s_%ishot_avg_%.2f.pth' % (args.backbone, args.shot, total_miou / 5)))



if __name__ == '__main__':
    main()
