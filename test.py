from rese import ReSe
from models.model import get_model

import argparse
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser("NAIC2020")
    parser.add_argument("--data-root", type=str, default="./image_B")
    parser.add_argument("--result-dir", type=str, default="./test_B")
    parser.add_argument("--log-dir", type=str, default="./result/resnet50_pspnet_001")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--model", type=str, default="pspnet")
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--tta", dest="tta", action="store_true")
    parser.add_argument("--save-mask", dest="save_mask", action="store_true")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    transforms = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    testset = ReSe(base_dir=args.data_root, split='test', transform=transforms)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=16, drop_last=False)
    model = get_model(args.model, args.backbone, args.pretrained, 
                    len(testset.CLASSES_LABEL))
    model.load_state_dict(torch.load(args.load_from), strict=True)
    model.eval()
    model = DataParallel(model).cuda()
    if not os.path.exists(os.path.join(args.result_dir, 'results')):
        os.makedirs(os.path.join(args.result_dir, 'results'))
    with torch.no_grad():
        tbar = tqdm(testloader)
        for img, id in tbar:
            img = img.cuda()
            predict = model(img, tta=True)
            predict = torch.argmax(predict, dim=1).cpu().numpy()
            predict = predict.astype(np.uint16)
            predict += 1
            predict *= 100
            for i, pred_mask in enumerate(predict):
                pred_mask = Image.fromarray(pred_mask)
                pred_mask.save(os.path.join(args.result_dir, 'results', id[i] + '.png'))
            tbar.set_description("Testing")