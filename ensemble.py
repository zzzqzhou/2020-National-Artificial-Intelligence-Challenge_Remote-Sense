from datasets.rese import ReSe
from models.model_zoo import get_model
from utils.palette import color_map

import numpy as np
import os
from PIL import Image
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser("Model Ensemble")
    parser.add_argument("--data-root", type=str, default="/home/ziqi/data/AI_Yaogan/image_B")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-1", type=str, default="resnet101")
    parser.add_argument("--model-1", type=str, default="deeplabv3plus")
    parser.add_argument("--backbone-2", type=str, default="resnext50")
    parser.add_argument("--model-2", type=str, default="deeplabv3plus")
    parser.add_argument("--load-from-1", type=str, default='/home/ziqi/data/AI_Yaogan/result/resnet101_deeplabv3plus_001/models/deeplabv3plus_resnet101_best.pth')
    parser.add_argument("--load-from-2", type=str ,default='/home/ziqi/data/AI_Yaogan/result/resnext50_deeplabv3plus_001/models/resnext50_deeplabv3plus_best.pth')
    parser.add_argument("--tta", dest="tta", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    args = parser.parse_args()
    print(args)
    transforms = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    testset = ReSe(base_dir=args.data_root, split='test', transform=transforms)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=16, drop_last=False)

    model_1 = get_model(args.model_1, args.backbone_1, args.pretrained, 
                    len(testset.CLASSES_LABEL))
    model_2 = get_model(args.model_2, args.backbone_2, args.pretrained, 
                    len(testset.CLASSES_LABEL))

    model_1.load_state_dict(torch.load(args.load_from_1), strict=True)
    model_2.load_state_dict(torch.load(args.load_from_2), strict=True)

    model_1.eval()
    model_2.eval()
    model_1 = DataParallel(model_1).cuda()
    model_2 = DataParallel(model_2).cuda()

    if not os.path.exists('test_B/results'):
        os.makedirs('test_B/results')
    
    with torch.no_grad():
        tbar = tqdm(testloader)
        for img, id in tbar:
            img = img.cuda()

            predict_1 = model_1(img, tta=True)
            predict_2 = model_2(img, tta=True)
            predict = (predict_1 + predict_2) / 2
            predict = torch.argmax(predict, dim=1).cpu().numpy()

            predict = predict.astype(np.uint16)
            predict += 1
            predict *= 100
            
            for i, pred_mask in enumerate(predict):
                pred_mask = Image.fromarray(pred_mask)
                pred_mask.save(os.path.join('test_B/results', id[i] + '.png'))
            
            tbar.set_description("Testing")