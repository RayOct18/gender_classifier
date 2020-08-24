import argparse
import better_exceptions
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from model.dex_models import Gender
from defaults import _C as cfg
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from dataloader import loadImgs


class ImageFolderWithPaths(dsets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        filename = os.path.split(path)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (filename[-1],))
        return tuple_with_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/media/ray/Ray/GoogleDrive/Avlab/program/Self-Attention-GAN-Experiment/generate/gan_morph_wgangp_z512_arcface_10llr_Din_flip_0001l2_0_1_stargan_GBN/test/fold0', help="Data root directory")
    parser.add_argument('--val_csv', type=str, default='./train_list/morph_test_fold0.csv', help='csv for validation')
    parser.add_argument("--save_path", type=str, default='result', help="Result path")
    parser.add_argument("--resume", type=str, default='./checkpoint/epoch012_0.00098_0.9947.pth', help="Result path")
    parser.add_argument("--version", type=str, default='model1', help="Result path")
    args = parser.parse_args()
    return args

def predict(validate_loader, model, device, args):

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    df = pd.DataFrame()
    filenames = []
    preds = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y, name) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)
                # compute output
                x = (x + 1) / 2
                x = x.clamp_(0, 1) * 255 # DEX
                outputs = model(x)
                filenames.append(name)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy().argmax(1))
    preds = np.concatenate(preds, axis=0)
    filenames = np.concatenate(filenames, axis=0)
    data = list(zip(filenames.tolist(), preds.tolist()))
    df = df.append(data, ignore_index=True)
    df.columns = ['name', 'gender']
    save_name = args.data_dir.split('/')
    df.to_csv(os.path.join(save_dir ,'{}_{}.csv'.format(*save_name[-3:-1])), index=False)

def main():
    args = get_args()

    cfg.freeze()

    # create model
    model = Gender()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(checkpoint)
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    # transform = transforms.Compose([transforms.Resize(256),
    #                                 transforms.RandomCrop(cfg.MODEL.IMG_SIZE),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # dataset = ImageFolderWithPaths(args.test_dir, transform=transform)

    # test_loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                           batch_size=cfg.TEST.BATCH_SIZE,
    #                                           shuffle=False,
    #                                           num_workers=cfg.TEST.WORKERS,
    #                                           drop_last=False)

    test_loader = loadImgs(dataset_path=args.data_dir, csv_dir=args.val_csv, img_size=cfg.MODEL.IMG_SIZE, 
                            batchSize=cfg.TEST.BATCH_SIZE, num_workers=cfg.TRAIN.WORKERS, shuffle=False, drop_last=False)


    best_val_acc = 0.0


    print("=> start predicting")
    predict(test_loader, model, device, args)
    print("Finish!")


if __name__ == '__main__':
    main()
