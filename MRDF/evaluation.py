import numpy as np
from scipy import stats
from sklearn import metrics
import torch

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        try:
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

            # Precisions, recalls
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                target[:, k], output[:, k])

            # FPR, TPR
            (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

            save_every_steps = 1000     # Sample statistics to reduce size
            dict = {
                # 'precisions': precisions[0::save_every_steps],
                    # 'recalls': recalls[0::save_every_steps],
                    'AP': avg_precision,
                    'fpr': fpr[0::save_every_steps],
                    'fnr': 1. - tpr[0::save_every_steps],
                    'auc': auc,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
        except Exception as e:
            dict = {'precisions': -1,
                    'recalls': -1,
                    'AP': avg_precision,
                    'fpr': -1,
                    'fnr': -1,
                    'auc': -1,
                    # note acc is not class-wise, this is just to keep consistent with other metrics
                    'acc': acc
                    }
            print('class {:s} no true sample'.format(str(k)))
        stats.append(dict)

    return stats
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
def run_evaluation(metadata, dict_video_label_pred, text):
    output = []
    target = []
    for sample in tqdm(metadata):
        label = dict_video_label_pred[sample['video_path']]['label']
        prediction = dict_video_label_pred[sample['video_path']]['prediction']
        output.append(prediction)
        target.append(np.zeros(2))
        target[-1][label] = 1
    target = np.array(target)
    output = np.array(output)
    print(text, calculate_stats(output, target))
if __name__ =="__main__":
    predictions = Dataset.load_from_disk("./predictions_pretrained")
    metadata = Dataset.load_from_disk("/home/fl488644/datasets/MAVOSSDD")
    dict_video_label_pred={}
    print(predictions[0])
    for pred in predictions:
        dict_video_label_pred[pred['video_path']] = {'label': pred['label'], "prediction": pred['logits']}
        # if dict_video_label_pred[pred['video_path']]['label'] ==0 and dict_video_label_pred[pred['video_path']]['label']!=dict_video_label_pred[pred['video_path']]['prediction']:
        #     print(pred['video_path'])
    print(metadata[0])
    metadata_indomain = metadata.filter(lambda sample: sample['split']=='test' and not sample['open_set_model'] and not sample['open_set_language'])
    metadata_open_model = metadata.filter(lambda sample: sample['split']=='test' and sample['open_set_model'])
    metadata_open_language = metadata.filter(lambda sample: sample['split']=='test' and sample['open_set_language'])
    metadata_all = metadata.filter(lambda sample: sample['split']=='test')
    run_evaluation(metadata_indomain, dict_video_label_pred, "Indomain")
    run_evaluation(concatenate_datasets([ metadata_indomain, metadata_open_model]), dict_video_label_pred, "Open_model")
    run_evaluation(concatenate_datasets([ metadata_indomain, metadata_open_language]), dict_video_label_pred, "Open_language")
    run_evaluation(metadata_all, dict_video_label_pred, "Open all")
