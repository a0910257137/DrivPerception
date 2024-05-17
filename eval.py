import argparse
import time
import numpy as np
from monitor import logger
from tqdm import tqdm
from pprint import pprint
from box import Box
import os
import sys
import cv2
import pandas as pd
from pathlib import Path
from utils.io import *
from utils.bdd_process import *
from metric.evaluate import ConfusionMatrix, SegmentationMetric, AverageMeter
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from DrivePredictor.inference import DrivePredictor


class Eval:

    def __init__(self, model, config, eval_path, img_root, batch_size):
        self.config = config
        self.pred_config = config['predictor']
        self.metric_config = config['metric']
        self.is_plot = self.pred_config["is_plot"]
        self.eval_path = eval_path
        self.img_root = img_root
        self.batch_size = batch_size
        self.mode = self.pred_config['predictor_mode']
        self.metric_type = self.metric_config['metric_type']
        self.img_input_size = self.pred_config.img_input_size
        self.predictor = model(config)
        self.nc = self.pred_config["nc"]
        self.confusion_matrix = ConfusionMatrix(nc=self.pred_config.nc)
        self.da_metric = SegmentationMetric(
            self.pred_config.num_seg_class)  #segment confusion matrix
        self.ll_metric = SegmentationMetric(2)  #segment confusion matrix
        self.da_acc_seg = AverageMeter()
        self.da_IoU_seg = AverageMeter()
        self.da_mIoU_seg = AverageMeter()
        self.ll_acc_seg = AverageMeter()
        self.ll_IoU_seg = AverageMeter()
        self.ll_mIoU_seg = AverageMeter()
        self.T_inf = AverageMeter()
        self.T_nms = AverageMeter()

    def get_eval_path(self):
        eval_files = []
        if os.path.isfile(self.eval_path):
            eval_files.append(self.eval_path)
        else:
            eval_files = [
                os.path.join(self.eval_path, x)
                for x in os.listdir(self.eval_path)
            ]
        return eval_files

    def split_batchs(self, elems, idx):
        model_imgs, resized_imgs, lb_gts, da_gts, ll_gts = [], [], [], [], []
        batch_frames = elems[idx:idx + self.batch_size]

        for elem in batch_frames:
            img_path = os.path.join(self.img_root, elem['name'])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR
                             | cv2.IMREAD_IGNORE_ORIENTATION)
            da_path = os.path.join(
                "/aidata/anders/autosys/annos/da_seg_annotations/imgs",
                elem['name'].replace(".jpg", ".png"))
            da_gt = cv2.imread(da_path, 0)
            lane_path = os.path.join(
                "/aidata/anders/autosys/annos/ll_seg_annotations/imgs",
                elem['name'].replace(".jpg", ".png"))
            ll_gt = cv2.imread(lane_path, 0)
            h0, w0 = img.shape[:2]
            resized_shape = np.max(self.img_input_size)
            r = resized_shape / max(h0, w0)
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                resized_img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                                         interpolation=interp)
                da_gt = cv2.resize(da_gt, (int(w0 * r), int(h0 * r)),
                                   interpolation=interp)
                ll_gt = cv2.resize(ll_gt, (int(w0 * r), int(h0 * r)),
                                   interpolation=interp)
            (resized_img, mask_img, lane_img), ratio, pad = self.letterbox(
                (resized_img, da_gt, ll_gt),
                int(resized_shape),
                auto=True,
                scaleup=True)
            h, w = mask_img.shape[:2]
            pad_w = int(pad[0])
            pad_h = int(pad[1])
            frame_kps = []
            frame_kps = [
                np.array([
                    o['box2d']['x1'], o['box2d']['y1'], o['box2d']['x2'],
                    o['box2d']['y2']
                ]) for o in elem['labels']
            ]
            det_labels = np.stack(frame_kps)
            nl = det_labels.shape[0]
            tcls = np.zeros(shape=(nl, 1))
            det_labels = np.concatenate([det_labels, tcls], axis=-1)
            lb_gts.append(det_labels)
            model_imgs.append(img)
            resized_imgs.append(resized_img)
            mask_img = mask_img[pad_h:h - pad_h, pad_w:w - pad_w]
            lane_img = lane_img[pad_h:h - pad_h, pad_w:w - pad_w]

            _, seg1 = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)
            _, seg2 = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY_INV)
            _, lane1 = cv2.threshold(lane_img, 1, 255, cv2.THRESH_BINARY)
            _, lane2 = cv2.threshold(lane_img, 1, 255, cv2.THRESH_BINARY_INV)
            seg1 = seg1 / 255.
            seg2 = seg2 / 255.
            lane1 = lane1 / 255.
            lane2 = lane2 / 255.
            mask_img = np.stack((seg2, seg1), axis=-1)
            mask_img = np.where(
                np.max(mask_img, axis=-1) == mask_img[..., 1], 1, 0)

            lane_img = np.stack((lane2, lane1), axis=-1)
            lane_img = np.where(
                np.max(lane_img, axis=-1) == lane_img[..., 1], 1, 0)
            da_gts.append(mask_img)
            ll_gts.append(lane_img)
        resized_imgs = np.stack(resized_imgs, axis=0)
        da_gts = np.stack(da_gts, axis=0)
        ll_gts = np.stack(ll_gts, axis=0)
        yield (model_imgs, resized_imgs, lb_gts, da_gts, ll_gts, batch_frames)

    def run(self):
        eval_files = self.get_eval_path()
        self.cates = load_text(self.pred_config.cat_path)
        output_dir = ""
        save_dir = output_dir + os.path.sep + 'visualization'
        iouv = np.linspace(0.5, 0.95, 10)  #iou vector for mAP@0.5:0.95
        niou = iouv.shape[0]

        if not os.path.join(save_dir):
            os.mkdir(save_dir)
        print("-" * 100)
        print('Eval categories {}'.format(self.cates))
        total_imgs = 0
        jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
        for eval_file in eval_files:
            print('Evaluating with %s' % eval_file)
            gt_bdd_annos = load_json(eval_file)
            gt_bdd_list = gt_bdd_annos['frame_list'][100:101]
            batch_objects = list(
                map(lambda x: self.split_batchs(gt_bdd_list, x),
                    range(0, len(gt_bdd_list), self.batch_size)))
            progress = tqdm(total=len(batch_objects))
            for batch_imgs_shapes in batch_objects:
                progress.update(1)
                batch_imgs_shapes = list(batch_imgs_shapes)
                total_imgs += len(batch_imgs_shapes)
                for imgs_shapes in batch_imgs_shapes:
                    model_imgs, _, lb_gts, da_gts, ll_gts, _ = imgs_shapes
                    batch_results = self.predictor.pred(model_imgs)
                    det_out, da_seg_out, ll_seg_out, b_color_area = batch_results
                    det_out, da_seg_out, ll_seg_out, b_color_area = det_out.numpy(
                    ), da_seg_out.numpy(), ll_seg_out.numpy(
                    ), b_color_area.numpy()
                    self.da_metric.reset()
                    self.da_metric.addBatch(da_seg_out, da_gts)
                    da_acc = self.da_metric.pixelAccuracy()
                    da_IoU = self.da_metric.IntersectionOverUnion()
                    da_mIoU = self.da_metric.meanIntersectionOverUnion()
                    self.da_acc_seg.update(da_acc, self.batch_size)
                    self.da_IoU_seg.update(da_IoU, self.batch_size)
                    self.da_mIoU_seg.update(da_mIoU, self.batch_size)

                    self.ll_metric.reset()
                    self.ll_metric.addBatch(ll_seg_out, ll_gts)
                    ll_acc = self.ll_metric.lineAccuracy()

                    ll_IoU = self.ll_metric.IntersectionOverUnion()
                    ll_mIoU = self.ll_metric.meanIntersectionOverUnion()
                    self.ll_acc_seg.update(ll_acc, self.batch_size)
                    self.ll_IoU_seg.update(ll_IoU, self.batch_size)
                    self.ll_mIoU_seg.update(ll_mIoU, self.batch_size)
                    seen = 0
                    for si, (preds, lb_gt) in enumerate(zip(det_out, lb_gts)):
                        mask = np.all(np.isfinite(preds), axis=-1)
                        preds = preds[mask]
                        nl = len(lb_gt)
                        tcls = lb_gt[:, -1] if nl else []  # target class
                        seen += 1
                        if len(preds) == 0:
                            if nl:
                                stats.append(
                                    (np.zeros(shape=(0, niou),
                                              dtype=np.bool_), [], [], tcls))
                            continue
                        # Assign all predictions as incorrect
                        correct = np.zeros(shape=(preds.shape[0], niou),
                                           dtype=bool)
                        if nl:
                            detected = []
                            tcls_tensor = lb_gt[:, -1]
                            for cls in np.unique(tcls_tensor):
                                ti = np.nonzero(cls == tcls_tensor)[0].reshape(
                                    -1)  # prediction indices

                                c = preds[:, -1]
                                pi = np.nonzero(cls == c)[0].reshape(
                                    -1)  # prediction indices
                                if pi.shape[0]:
                                    # Prediction to target ious
                                    # n*m  n:pred  m:label
                                    ious = self.box_iou(
                                        preds[pi, :4], lb_gt[ti, :4])
                                    max_ious = np.max(ious, axis=-1)
                                    i = np.argmax(ious, axis=-1)
                                    # Append detections
                                    detected_set = set()
                                    for j in np.nonzero(max_ious > iouv[0])[0]:
                                        d = ti[i[j]]  # detected target
                                        if d not in detected_set:
                                            detected_set.add(d)
                                            detected.append(d)
                                            correct[pi[j]] = max_ious[
                                                j] > iouv  # iou_thres is 1xn
                                            if len(
                                                    detected
                                            ) == nl:  # all targets already located in image
                                                break
                        # Append statistics (correct, conf, pcls, tcls)
                        stats.append((correct, preds[:, 4], preds[:, 5], tcls))
        # Compute statistics
        # stats : [[all_img_correct]...[all_img_tcls]]
        stats = [np.concatenate(x, 0)
                 for x in zip(*stats)]  # to numpy  zip(*) :unzip
        map70, map75 = None, None
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = self.ap_per_class(*stats,
                                                       plot=False,
                                                       save_dir=save_dir)

            ap50, ap70, ap75, ap = ap[:, 0], ap[:, 4], ap[:, 5], ap.mean(
                1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map70, map75, maps = p.mean(), r.mean(), ap50.mean(
            ), ap70.mean(), ap75.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64),
                             minlength=self.nc)  # number of targets per class
        else:
            nt = np.zeros(1)

        verbose = False
        # Print results
        pf = '%10s' + '%12.3g' * 6  # print format
        # Print results per class
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, maps))

        if (verbose or (self.nc <= 20)) and self.nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        maps = np.zeros(self.nc) + maps
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        da_segment_result = (self.da_acc_seg.avg, self.da_IoU_seg.avg,
                             self.da_mIoU_seg.avg)
        ll_segment_result = (self.ll_acc_seg.avg, self.ll_IoU_seg.avg,
                             self.ll_mIoU_seg.avg)
        detect_result = np.asarray([mp, mr, map50, maps])
        # print('mp:{},mr:{},map50:{},map:{}'.format(mp, mr, map50, map))
        #print segmet_result
        print(
            '-----------------------------Finish evaluating-----------------')
        return da_segment_result, ll_segment_result, detect_result, maps

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])  #(x2-x1)*(y2-y1)

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) -
                 np.maximum(box1[:, None, :2], box2[:, :2]))
        inter = np.clip(inter, a_min=0, a_max=np.inf)
        inter = np.prod(inter, axis=2, dtype=np.float32)
        return inter / (area1[:, None] + area2 - inter
                        )  # iou = inter / (area1 + area2 - inter)

    def ap_per_class(self,
                     tp,
                     conf,
                     pred_cls,
                     target_cls,
                     plot=False,
                     save_dir='precision-recall_curve.png',
                     names=[]):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            save_dir:  Plot save directory
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
        s = [
            unique_classes.shape[0], tp.shape[1]
        ]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
        ap, p, r = np.zeros(s), np.zeros(
            (unique_classes.shape[0], 1000)), np.zeros(
                (unique_classes.shape[0], 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = (target_cls == c).sum()  # number of labels
            n_p = i.sum()  # number of predictions

            if n_p == 0 or n_l == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_l + 1e-16)  # recall curve
                r[ci] = np.interp(
                    -px, -conf[i], recall[:, 0],
                    left=0)  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p[ci] = np.interp(-px, -conf[i], precision[:, 0],
                                  left=1)  # p at pr_score
                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = self.compute_ap(
                        recall[:, j], precision[:, j])

                    if plot and (j == 0):
                        py.append(np.interp(px, mrec,
                                            mpre))  # precision at mAP@0.5

        # Compute F1 score (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + 1e-16)
        i = r.mean(0).argmax()

        if plot:
            self.plot_pr_curve(px, py, ap, save_dir, names)

        return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.], recall, [recall[-1] + 1E-3]))
        mpre = np.concatenate(([1.], precision, [0.]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[
                0]  # points where x axis (recall) changes
            ap = np.sum(
                (mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    def plot_pr_curve(self, px, py, ap, save_dir='.', names=()):
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        py = np.stack(py, axis=1)

        if 0 < len(names) < 21:  # show mAP in legend if < 10 classes
            for i, y in enumerate(py.T):
                ax.plot(px,
                        y,
                        linewidth=1,
                        label=f'{names[i]} %.3f' %
                        ap[i, 0])  # plot(recall, precision)
        else:
            ax.plot(px, py, linewidth=1,
                    color='grey')  # plot(recall, precision)

        ax.plot(px,
                py.mean(1),
                linewidth=3,
                color='blue',
                label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.savefig(Path(save_dir) / 'precision_recall_curve.png', dpi=250)

    def letterbox(self,
                  combination,
                  new_shape=(640, 640),
                  color=(114, 114, 114),
                  auto=True,
                  scaleFill=False,
                  scaleup=True):
        """Resize the input image and automatically padding to suitable shape :https://zhuanlan.zhihu.com/p/172121380"""
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        img, gray, line = combination
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
            1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[
                0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            gray = cv2.resize(gray, new_unpad, interpolation=cv2.INTER_LINEAR)
            line = cv2.resize(line, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img,
                                 top,
                                 bottom,
                                 left,
                                 right,
                                 cv2.BORDER_CONSTANT,
                                 value=color)  # add border
        gray = cv2.copyMakeBorder(gray,
                                  top,
                                  bottom,
                                  left,
                                  right,
                                  cv2.BORDER_CONSTANT,
                                  value=0)  # add border
        line = cv2.copyMakeBorder(line,
                                  top,
                                  bottom,
                                  left,
                                  right,
                                  cv2.BORDER_CONSTANT,
                                  value=0)  # add border
        # print(img.shape)

        combination = (img, gray, line)
        return combination, ratio, (dw, dh)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--config', default=None, help='eval config')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--eval_path',
                        default=None,
                        help='eval data folder or file path')
    parser.add_argument('--img_root', help='eval images folder path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('Eval with %s' % args.config)
    if not os.path.isfile(args.config):
        raise FileNotFoundError('File %s does not exist.' % args.config)
    config = load_json(args.config)
    eval = Eval(DrivePredictor, Box(config), args.eval_path, args.img_root,
                args.batch_size)
    eval.run()
