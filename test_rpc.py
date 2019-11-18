import sys


sys.path.append('.')
sys.path.append('..')
from lib.networks.model_repository import *
from lib.datasets.checkout import CheckoutDetection, CHECKOUT_ROOT
import torch
import time
import argparse
import cv2
import random
from tools.utils import ctdet_decode


parser = argparse.ArgumentParser(
    description='Checkout Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=CHECKOUT_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--trained_model', default='weights/CountNet_epoch_25_steps_4500_0.002534.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("cuda可以用")
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    print("cuda不能用")


def test_net(net, cuda, testset, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = 'result.json'
    num_images = len(testset)
    result = list()
    for i in range(num_images):
        # print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        index = random.randint(0, len(testset))
        img_path, labels, boxes = testset.pull_annotation(index)    # 随机
        testset.__getitem__(index)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        img = cv2.resize(img, (testset.size, testset.size))
        boxes[:, (0, 2)] = boxes.astype(float)[:, (0, 2)] // (h / testset.size)
        boxes[:, (1, 3)] = boxes.astype(float)[:, (1, 3)] // (w / testset.size)

        testset.visualize_bbox(img, labels, boxes)
        x = testset.base_transform(img)

        if cuda and torch.cuda.is_available():
            x = x.cuda()

        x = x.unsqueeze(0)    # one image a batch
        print("开始预测")
        with torch.no_grad():
            y = net(x)  # forward pass

        # 细粒度问题，长得差不多的类别会错误识别
        # 太小的物体也会识别失败        
        heatmap = y.cpu()
        """
        print(heatmap.shape, time.time()-t1, heatmap.max())
        for label in labels:
            print(label, heatmap[label].max())
            cv2.imshow(str(label), heatmap[label])
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        """
        start_time = time.time()
        dets = ctdet_decode(heatmap, K=50)    # # shape=(batch, K, 4)    center_points, scores, clses
        # forward_time = time.time()-start_time
        testset.debug([img, ], dets)
        cv2.destroyAllWindows()


def test_checkout():
    # load net
    device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
    print(device)
    model = Resnet18_8s(ver_dim=201)  # 这是训练一个类别的
    model.to(device)
    model.load_state_dict(torch.load(args.trained_model, map_location=lambda storage, location: storage))
    model.eval()
    print('Finished loading model!')
    # load data
    testset = CheckoutDetection(CHECKOUT_ROOT, 'val', show_images=True)
    # evaluation
    test_net(model, args.cuda, testset, thresh=args.visual_threshold)


if __name__ == "__main__":
    test_checkout()
