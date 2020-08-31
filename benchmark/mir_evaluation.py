import numpy as np
import pandas as pd

from mir_eval import transcription

def evaluate(ref_csv, pred_csv):
    ref_intervals_start = [int(x) for x in list((ref_csv['start_time'] - ref_csv['start_time'].min()).values)]
    ref_intervals_end = [int(x) for x in list((ref_csv['end_time'] - ref_csv['start_time'].min()).values)]
    ref_intervals = np.array([ref_intervals_start, ref_intervals_end]).T

    pred_intervals_start = [int(x) for x in list((pred_csv['onset_time'] - pred_csv['onset_time'].min()).values)]
    pred_intervals_end = [int(x) for x in list((pred_csv['offset_time'] - pred_csv['onset_time'].min()).values)]
    pred_intervals = np.array([pred_intervals_start, pred_intervals_end]).T

    ref_pitches = np.array([int(x) for x in list(ref_csv['note'].values)])
    pred_pitches = np.array([int(x) for x in list(pred_csv['pitch'].values)])

    return transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            pred_intervals,
            pred_pitches)

ref_csv = pd.read_csv("musicnet/test_labels/1759.csv")
pred_csv = pd.read_csv("googlebrain/1759.csv")
print(evaluate(ref_csv, pred_csv))