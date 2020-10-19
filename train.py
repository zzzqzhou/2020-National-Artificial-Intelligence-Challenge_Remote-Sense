from rese import ReSe
from models.model import get_model

import argparse
import os
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from tqdm import tqdm
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("NAIC2020")
    parser.add_argument("--data-root", type=str, default="/home/ziqi/data/AI_Yaogan/train")
    parser.add_argument("--log-dir", type=str, default="/home/ziqi/data/AI_Yaogan/result/resnet50_deeplabv3plus")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--model", type=str, default="deeplabv3plus")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    transforms = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trainset = ReSe(base_dir=args.data_root, split='train', transform=transforms)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16, drop_last=True)
        
    model = get_model(args.model, args.backbone, args.pretrained, len(trainset.CLASSES_LABEL))

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from), strict=True)
        
    criterion = CrossEntropyLoss()

    optimizer = SGD([{"params": model.backbone.parameters(), "lr": args.lr},
                     {"params": [param for name, param in model.named_parameters()
                                if "backbone" not in name],
                      "lr": args.lr * 10.0}],
                    lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
    model = DataParallel(model).cuda()
    iters = 0
    total_iters = len(trainloader) * args.epochs
    previous_best = 0.0
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, 'models')):
        os.makedirs(os.path.join(args.log_dir, 'models'))

    train_log = open(os.path.join(args.log_dir, 'train.log'), 'w')
    for epoch in range(args.epochs):
        print("\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.2f" %
              (epoch + 1, optimizer.param_groups[0]["lr"], previous_best))
        tbar = tqdm(trainloader)
        model.train()
        total_loss = 0.0
        count = 0
        for i, (img, mask) in enumerate(tbar):
            count += 1
            img, mask = img.cuda(), mask.cuda()

            predict = model(img)
            loss = criterion(predict, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10.0
            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))
        train_loss = total_loss / count
        train_log.write('Epoch: %d, Train Loss: %.3f\n' % (epoch + 1, train_loss))
        train_log.flush()
        torch.save(model.module.state_dict(), os.path.join(args.log_dir, 'models', '%s_%s_best.pth' % (args.backbone, args.model)))
    train_log.close()