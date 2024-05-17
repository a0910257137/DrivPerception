import numpy as np
import copy
import pandas as pd
from pprint import pprint

kp_base_dict = {
    "left_eye_lnmk_27": None,
    "right_eye_lnmk_33": None,
    "nose_lnmk_42": None,
    "outer_lip_lnmk_48": None,
    "outer_lip_lnmk_54": None
}


def offset_v2_to_tp_od_bdd(bdd_results, batches_preds, batches_frames, cates):
    b_bboxes, b_lnmks, b_nose_scores = batches_preds
    b_bboxes, b_lnmks, b_nose_scores = b_bboxes.numpy(), b_lnmks.numpy(
    ), b_nose_scores.numpy()
    for bboxes, lnmks, nose_scores, frames in zip(b_bboxes, b_lnmks,
                                                  b_nose_scores,
                                                  batches_frames):

        pred_frame = {
            'dataset': frames['dataset'],
            'sequence': frames['sequence'],
            'name': frames['name'],
            'labels': []
        }
        valid_mask = np.all(np.isfinite(bboxes), axis=-1)
        bboxes = bboxes[valid_mask]
        hws = bboxes[:, 2:4] - bboxes[:, :2]
        areas = hws[:, 0] * hws[:, 1]
        n_bbox = [[]]
        n_lnmk = [[None] * 5]
        if len(areas) != 0:
            idx = np.argmax(areas, axis=0)
            bbox = bboxes[idx]
            tl, br = bbox[:2], bbox[2:4]
            y1, x1 = tl
            y2, x2 = br

            nose_lnmks = lnmks[:, 2, :]
            logical_y = np.logical_and(y1 < nose_lnmks[:, :1],
                                       nose_lnmks[:, :1] < y2)
            logical_x = np.logical_and(x1 < nose_lnmks[:, 1:],
                                       nose_lnmks[:, 1:] < x2)
            logical_yx = np.concatenate([logical_y, logical_x], axis=-1)
            inside_masks = np.all(logical_yx, axis=-1)
            nose_lnmks = nose_lnmks[inside_masks]
            nose_scores = nose_scores[inside_masks]
            if len(nose_scores) != 0:
                max_idx = np.argmax(nose_scores)
                n_bbox = np.reshape(bbox, (-1, 6))
                n_lnmk = np.reshape(lnmks[max_idx], (-1, 5, 2))
        for bbox, lnmk in zip(n_bbox, n_lnmk):
            if len(bbox) != 0:
                pred_lb = {
                    'category': cates[int(bbox[5])].upper(),
                    'box2d': {
                        'y1': float(bbox[0]),
                        'x1': float(bbox[1]),
                        'y2': float(bbox[2]),
                        'x2': float(bbox[3])
                    }
                }
                kp_base = copy.deepcopy(kp_base_dict)
                if not np.all(lnmk == None):
                    kp_base['left_eye_lnmk_27'] = lnmk[0].tolist()
                    kp_base['right_eye_lnmk_33'] = lnmk[1].tolist()
                    kp_base['nose_lnmk_42'] = lnmk[2].tolist()
                    kp_base['outer_lip_lnmk_48'] = lnmk[3].tolist()
                    kp_base['outer_lip_lnmk_54'] = lnmk[4].tolist()
                pred_lb['keypoints'] = kp_base
                pred_frame['labels'].append(pred_lb)
        bdd_results['frame_list'].append(pred_frame)
    return bdd_results


def to_tp_od_bdd(bdd_results, batches_preds, batches_frames, cates):

    for preds, frames in zip(batches_preds, batches_frames):
        pred_frame = {
            'dataset': frames['dataset'],
            'sequence': frames['sequence'],
            'name': frames['name'],
            'labels': []
        }
        preds = preds.numpy()
        valid_mask = np.all(~np.isinf(preds), axis=-1)
        preds = preds[valid_mask]
        for pred_obj, frame_lb in zip(preds, frames["labels"]):
            gt_lnmks = frame_lb['keypoints']
            keys = gt_lnmks.keys()

            eval_lnmks = copy.deepcopy(gt_lnmks)
            for key in keys:
                gt_lnmks[key] = None
                eval_lnmks[key] = None
                if key == 'left_eye_lnmk_27':
                    eval_lnmks[key] = pred_obj[6:8]
                elif key == 'right_eye_lnmk_33':
                    eval_lnmks[key] = pred_obj[8:10]
                elif key == 'nose_lnmk_42':
                    eval_lnmks[key] = pred_obj[10:12]
                elif key == 'outer_lip_lnmk_48':
                    eval_lnmks[key] = pred_obj[12:14]
                elif key == 'outer_lip_lnmk_54':
                    eval_lnmks[key] = pred_obj[14:16]
            pred_lb = {
                'keypoints': eval_lnmks,
                'category': cates[int(pred_obj[5])].upper(),
                'box2d': {
                    'y1': float(pred_obj[0]),
                    'x1': float(pred_obj[1]),
                    'y2': float(pred_obj[2]),
                    'x2': float(pred_obj[3])
                }
            }

            pred_frame['labels'].append(pred_lb)
        bdd_results['frame_list'].append(pred_frame)
    return bdd_results


def to_lnmk_bdd(bdd_results, batches_preds, batches_frames, lnmk_scheme):
    batches_preds = batches_preds.numpy()
    for preds, frame in zip(batches_preds, batches_frames):
        pred_frame = {
            'dataset': frame['dataset'],
            'sequence': frame['sequence'],
            'name': frame['name'],
            'labels': []
        }
        preds = preds.astype(np.float32)
        LE, RE, N, LM, RM = np.mean(preds[3:9], axis=0), np.mean(
            preds[9:15], axis=0), preds[16], preds[17], preds[21]
        preds = np.stack([LE, RE, N, LM, RM])
        for gt_lnmks in frame['labels']:
            pred_lb = copy.deepcopy(gt_lnmks['keypoints'])
            keys = list(pred_lb.keys())
            for key, pred_kp in zip(keys, preds):
                pred_lb[key] = pred_kp.tolist()

            pred_frame['labels'].append({
                "keypoints": pred_lb,
                "box2d": [],
                "category": "FACE"
            })
        bdd_results['frame_list'].append(pred_frame)
    return bdd_results


def tdmm_to_bdd(bdd_results, batches_preds, batches_frames, cates):

    b_bboxes, b_lnmks, b_poses = batches_preds[0].numpy(
    ), batches_preds[1].numpy(), batches_preds[2].numpy()

    for bboxes, lnmks, poses, frames in zip(b_bboxes, b_lnmks, b_poses,
                                            batches_frames):
        pred_frame = {
            'dataset': frames['dataset'],
            'sequence': frames['sequence'],
            'name': frames['name'],
            'labels': []
        }
        valid_mask = np.all(np.isfinite(bboxes), axis=-1)
        bboxes = bboxes[valid_mask]
        valid_mask = np.all(np.isfinite(lnmks), axis=-1)
        lnmks = np.reshape(lnmks[valid_mask], (-1, 68, 2))

        valid_mask = np.all(np.isfinite(poses), axis=-1)
        poses = np.reshape(poses[valid_mask], (-1, 3))

        lnmk_keys = frames["labels"][0]["keypoints"].keys()
        for bbox, lnmk, pose in zip(bboxes, lnmks, poses):
            pred_lb = {
                'category': cates[int(bbox[5])].upper(),
                'box2d': {
                    'y1': float(bbox[0]),
                    'x1': float(bbox[1]),
                    'y2': float(bbox[2]),
                    'x2': float(bbox[3])
                },
                'pose': {
                    "pitch": pose[0],
                    "yaw": pose[1],
                    "roll": pose[2]
                },
                "keypoints": {
                    k: lnmk[i].tolist()
                    for i, k in enumerate(lnmk_keys)
                }
            }
            pred_frame['labels'].append(pred_lb)
        bdd_results['frame_list'].append(pred_frame)
    return bdd_results


def scrfd_tdmm_to_bdd(bdd_results, batches_preds, batches_frames, cates):
    b_bboxes, b_lnmks = batches_preds[0].numpy(), batches_preds[1].numpy()
    for bboxes, lnmks, frames in zip(b_bboxes, b_lnmks, batches_frames):
        valid_mask = np.all(np.isfinite(bboxes), axis=-1)
        bboxes = bboxes[valid_mask]
        valid_mask = np.all(np.isfinite(lnmks), axis=-1)
        lnmks = lnmks[valid_mask].reshape((-1, 68, 2))
        lnmk_keys = frames["labels"][0]["keypoints"].keys()
        pred_frame = {
            'dataset': frames['dataset'],
            'sequence': frames['sequence'],
            'name': frames['name'],
            'labels': []
        }

        # print(frames["labels"][0])
        for bbox, lnmk in zip(bboxes, lnmks):

            # box_centers = (bbox[:2] + bbox[2:4]) / 2
            # lnmk_centers = (np.min(lnmk, axis=0) + np.max(lnmk, axis=0)) / 2
            # shifts = box_centers - lnmk_centers
            # bbox = bbox.astype(np.float32)
            # lnmk = lnmk + shifts[None, :]
            lnmk = lnmk.astype(np.float32)
            cat_num = 0
            pred_lb = {
                'category': cates[cat_num].upper(),
                'box2d': {
                    'y1': float(bbox[0]),
                    'x1': float(bbox[1]),
                    'y2': float(bbox[2]),
                    'x2': float(bbox[3])
                },
                "keypoints": {
                    k: lnmk[i].astype(np.float32).tolist()
                    for i, k in enumerate(lnmk_keys)
                }
            }
            pred_frame['labels'].append(pred_lb)
        bdd_results['frame_list'].append(pred_frame)
    return bdd_results


def scrfd_to_bdd(bdd_results, batches_preds, batches_frames, cates):
    b_bboxes = batches_preds.numpy()
    for bboxes, frames in zip(b_bboxes, batches_frames):
        valid_mask = np.all(np.isfinite(bboxes), axis=-1)
        bboxes = bboxes[valid_mask]
        pred_frame = {
            'dataset': frames['dataset'],
            'sequence': frames['sequence'],
            'name': frames['name'],
            'labels': []
        }
        for bbox in bboxes:
            pred_lb = {
                'category': cates[int(bbox[5])].upper(),
                'box2d': {
                    'y1': float(bbox[0]),
                    'x1': float(bbox[1]),
                    'y2': float(bbox[2]),
                    'x2': float(bbox[3])
                }
            }
            pred_frame['labels'].append(pred_lb)
        bdd_results['frame_list'].append(pred_frame)
    return bdd_results


def transform_pd_data(report_results,
                      is_post_cal,
                      mode,
                      metric_type='keypoints',
                      eval_types=('accuracy', 'precision', 'recall')):
    mean_df = None
    if metric_type.lower() == 'keypoints' or metric_type.lower(
    ) == 'landmarks':
        # pd_headers = {'center_point': [], 'top_left': [], 'bottom_right': []}
        # pd_headers = {
        #     'left_eye_lnmk_27': [],
        #     'right_eye_lnmk_33': [],
        #     'nose_lnmk_42': [],
        #     'outer_lip_lnmk_48': [],
        #     'outer_lip_lnmk_54': []
        # }
        pd_headers = {
            'countour': [],
            'left_eye': [],
            'right_eye': [],
            'nose': [],
            'lip': []
        }
    elif metric_type == 'IoU' or metric_type.lower() == 'box2d':
        pd_headers = {'IoU': []}
    data_frame = {'Eval_type': eval_types}

    for t in eval_types:
        type_results = report_results[t]
        if mode == 'tdmm' or mode == 'scrfd_tdmm':
            for k in pd_headers.keys():
                tmp = []
                for result_key in type_results.keys():
                    if k in result_key:
                        tmp.append(type_results[result_key])
                pd_headers[k].append(np.mean(tmp))

        else:
            for key in pd_headers:
                if key in list(type_results.keys()):
                    eval_vals = type_results[key]
                    pd_headers[key].append(eval_vals)
    if is_post_cal:
        post_mean_infos = {
            'countour': [],
            'left_eye': [],
            'right_eye': [],
            'nose': [],
            'lip': []
        }
        for pd_key in pd_headers:
            for post_key in post_mean_infos:
                if post_key in pd_key:
                    post_mean_infos[post_key].append(pd_headers[pd_key])
        for post_key in post_mean_infos:
            array = np.stack(post_mean_infos[post_key])
            array = np.mean(array, axis=0)
            post_mean_infos[post_key] = array
        mean_data_frame = copy.deepcopy(data_frame)
        mean_data_frame.update(
            {k: post_mean_infos[k]
             for k in post_mean_infos})
        print(mean_data_frame)
        mean_df = pd.DataFrame(mean_data_frame)

    data_frame.update({k: pd_headers[k] for k in pd_headers})
    df = pd.DataFrame(data_frame)
    return df, mean_df
