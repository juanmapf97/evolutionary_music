#!/usr/bin/env python
# coding: utf-8
import librosa
import numpy as np
import pandas as pd
import sys
import tensorflow.compat.v1 as tf
import time
import warnings

from magenta.common import tf_utils
from note_seq import audio_io
from magenta.models.onsets_frames_transcription import audio_label_data_utils
from magenta.models.onsets_frames_transcription import configs
from magenta.models.onsets_frames_transcription import constants
from magenta.models.onsets_frames_transcription import data
from magenta.models.onsets_frames_transcription import infer_util
from magenta.models.onsets_frames_transcription import train_util
import note_seq
from note_seq import midi_io
from note_seq import sequences_lib
from os import listdir
from os.path import isfile, join

MAESTRO_CHECKPOINT_DIR = 'benchmark/googlebrain/train'
tf.disable_v2_behavior()

model_type = "MAESTRO (Piano)"

if model_type.startswith('MAESTRO'):
  config = configs.CONFIG_MAP['onsets_frames']
  hparams = config.hparams
  hparams.use_cudnn = False
  hparams.batch_size = 1
  checkpoint_dir = MAESTRO_CHECKPOINT_DIR
# elif model_type.startswith('E-GMD'):
#   config = configs.CONFIG_MAP['drums']
#   hparams = config.hparams
#   hparams.batch_size = 1
#   checkpoint_dir = EGMD_CHECKPOINT_DIR
else:
  raise ValueError('Unknown Model Type')

examples = tf.placeholder(tf.string, [None])

dataset = data.provide_batch(
    examples=examples,
    preprocess_examples=True,
    params=hparams,
    is_training=False,
    shuffle_examples=False,
    skip_n_initial_records=0)

estimator = train_util.create_estimator(
    config.model_fn, checkpoint_dir, hparams)

iterator = dataset.make_initializable_iterator()
next_record = iterator.get_next()

def transcribe_audio(data_path="maps/test_data", transcription_path="data/transcription/googlebrain/"):
    fns = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]

    duration_dataset = {
        'song': [],
        'inference_time_in_seconds': []
    }

    for fn in fns:
        print(fn)
        with open(fn, 'rb') as fd:
            x = fd.read()
        
        to_process = []
        now = time.time()
        example_list = list(
        audio_label_data_utils.process_record(
            wav_data=x,
            ns=note_seq.NoteSequence(),
            example_id=fn,
            min_length=0,
            max_length=-1,
            allow_empty_notesequence=True))
        assert len(example_list) == 1
        to_process.append(example_list[0].SerializeToString())

        sess = tf.Session()

        sess.run([
            tf.initializers.global_variables(),
            tf.initializers.local_variables()
        ])

        sess.run(iterator.initializer, {examples: to_process})

        def transcription_data(params):
            del params
            return tf.data.Dataset.from_tensors(sess.run(next_record))
        input_fn = infer_util.labels_to_features_wrapper(transcription_data)

        prediction_list = list(
            estimator.predict(
                input_fn,
                yield_single_examples=False))
        assert len(prediction_list) == 1

        sequence_prediction = note_seq.NoteSequence.FromString(
            prediction_list[0]['sequence_predictions'][0])

        # Ignore warnings caused by pyfluidsynth
        warnings.filterwarnings("ignore", category=DeprecationWarning) 

        midi_filename = (transcription_path + fn.split(".")[0].split("/")[-1] + '.mid')
        csv_filename = (transcription_path + fn.split(".")[0].split("/")[-1] + '.csv')

        song_df_dict = {
            'onset_time': [],
            'offset_time': [],
            'pitch': []
        }
        for n in sequence_prediction.notes:
            song_df_dict['onset_time'].append(n.start_time * 1000)
            song_df_dict['offset_time'].append(n.end_time * 1000)
            song_df_dict['pitch'].append(n.pitch)

        midi_io.sequence_proto_to_midi_file(sequence_prediction, midi_filename)
        song_df = pd.DataFrame(song_df_dict)
        song_df.to_csv(csv_filename)
        
        now_now = time.time()
        
        duration_dataset['song'].append(fn.split("/")[-1])
        duration_dataset['inference_time_in_seconds'].append(now_now - now)

    df = pd.DataFrame(duration_dataset)
    df.to_csv(transcription_path + 'maps_times.csv')

if __name__ == "__main__":
    if len(sys.argv) == 3:
        transcribe_audio(sys.argv[1], sys.argv[2])
    else:
        transcribe_audio()