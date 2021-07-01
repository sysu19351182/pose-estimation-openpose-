import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

def zunfanrun_video(vedioname, prehumans=[]):
    logger = logging.getLogger('TfPoseEstimator-WebCam')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fps_time = 0

    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default=0)
    # 256*256较为合适，数字太小虽然处理速度变快但是效果一般
    parser.add_argument('--resize', type=str, default='256x256',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    # 模型有help里面的种类，大模型的处理速度会很慢，开始预训练使用mobilenet_thin就可以
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:

        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    vediopath = vedioname+'.mp4'
    cam = cv2.VideoCapture(vediopath)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    fps = cam.get(cv2.CAP_PROP_FPS)  # 视频帧率
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # print(image.shape[0],image.shape[1])
    frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    savepath = vedioname+'.avi'
    videoWriter = cv2.VideoWriter(savepath, fourcc, fps, frame_size)
    while ret_val:

        print(image.shape)
        print('-------')
        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        # emptyImage = np.zeros(image.shape, np.uint8)
        # emptyImage[...] = 0
        # 注释掉这两行代码以及下面参数有emptyImage改为image可实现背景是原来的视频
        pose_img = TfPoseEstimator.draw_humans(image, humans, prehumans, imgcopy=False)

        logger.debug('show+')

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.imshow('tf-pose-estimation result', pose_img)
        videoWriter.write(pose_img)
        fps_time = time.time()
        ret_val, image = cam.read()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    return humans
    cv2.destroyAllWindows()

