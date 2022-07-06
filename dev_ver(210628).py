from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.04, type=float, help='visualization_threshold')

# parser.add_argument('-m', '--trained_model', default='..\insightface\\retinaface\weights\mobilenet0.25_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
# parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
# parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

args = parser.parse_args()


def anonymize_face_simple(image, factor=3.0):
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1
    # apply a Gaussian blur to the input image using our computed
    # kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)


def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)
    # return the pixelated blurred image
    return image


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, "cuda")
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # params
    resize = 1
    black = (0, 0, 0)
    count = 0
    old_det = []
    _list = []
    _c = 0
    temp_ = []

    # (수정예정) 기존에 가져온 파일경로에 맞는 영상 생성
    VIDEO_FILE_PATH = "data/test5.mp4"
    OUTPUT_FILE = 'data/Processed4_10.mp4'

    # 비디오 불러오기
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if cap.isOpened() == False:
        print("Can\'t open the video(%d)" % (VIDEO_FILE_PATH))
        exit()

    starttime = time.time()
    # 재생할 파일의 넓이 얻기
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 재생할 파일의 높이 얻기
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 재생할 파일의 FPS 얻기
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('width {0}, height {1}, fps {2}'.format(width, height, fps))
    # 경로 수정

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (int(width), int(height)))
    print("Video Loaded... Processing Start")

    # 실질적인 start
    while True:
        # initialized params
        count += 1
        cur_list = []

        print(count, 'frame processing..')
        ret, frame = cap.read ()

        if frame is None:
            print("Video Converted Completed!")
            print("Total Elapsed Time:", int(time.time() - starttime) / 60, 'min')
            break

        # Face Detection
        img_raw = frame
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])

        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)

        # show image
        i=0
        if args.save_image:

            for b in dets:
                i += 1

                if b[4] < args.vis_thres:
                    continue
                cur_list.append([b[0], b[1], b[2], b[3]])

                try:
                    x1, x2, y1, y2 = int(b[0]), int(b[2]), int(b[1]), int(b[3])
                    # print('좌표는, ', x1, x2, y1, y2)
                    face = img_raw[y1:y2, x1:x2]  # 공통 영역
                    #pixelate 처리
                    face = anonymize_face_pixelate(face, blocks=3)
                    img_raw[y1:y2, x1:x2] = face
                    a = 50
                # #     # 검은색 처리
                #     cv2.rectangle(img_raw, (x1 - a, y1 - a), (x2 + a, y2 + a), black, -1)
                #
                #
                #
                except:
                    continue



            if len(cur_list) > len(_list):

                for bb in cur_list:
                    _list = cur_list
                    i += 1
                    try:
                        x1, x2, y1, y2 = int(bb[0]), int(bb[2]), int(bb[1]), int(bb[3])
                        # print('좌표는, ', x1, x2, y1, y2)
                        face = img_raw[y1:y2, x1:x2] # 공통 영역
                        # pixelate 처리
                        face = anonymize_face_pixelate(face, blocks=3)
                        img_raw[y1:y2, x1:x2] = face
                        a = 50
                        # # 검은색 처리
                        # cv2.rectangle(img_raw, (x1-a, y1-a), (x2+a, y2+a), black, -1)


                    except:
                        continue

                out.write(img_raw)
                print('out...', count)
            # elif len(cur_list) < len(_list) and _c < 20:
            elif len(cur_list) < len(_list) and _c < 20:

                for box in _list:
                    i += 1
                    try:
                        x1, x2, y1, y2 = int(box[0]), int(box[2]), int(box[1]), int(box[3])
                        print('residual coor ', x1, x2, y1, y2)
                        face = img_raw[x1:x2, y1:y2]
                        face = anonymize_face_pixelate(face, blocks=3)
                        img_raw[x1:x2, y1:y2] = face
                        a = 50
                        # # 검은색 처리
                        # cv2.rectangle(img_raw, (x1-a, y1-a), (x2+a, y2+a), black, -1)



                    except:
                        continue


                _c += 1
                out.write(img_raw)
                print('out...', count)

            else:
                _c = 0
                _list = []


    cap.release()
    out.release()
    cv2.destroyAllWindows()
