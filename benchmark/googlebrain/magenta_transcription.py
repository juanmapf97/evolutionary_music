#!/usr/bin/env python
# coding: utf-8

# In[3]:

import pandas as pd

import librosa


# In[4]:


MAESTRO_CHECKPOINT_DIR = 'googlebrain/train'


# In[5]:


import tensorflow.compat.v1 as tf


# In[6]:


import numpy as np


# In[7]:


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


# In[8]:


tf.disable_v2_behavior()


# In[9]:


model_type = "MAESTRO (Piano)"


# In[10]:


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


# In[11]:


from os import listdir
from os.path import isfile, join
import time


# In[ ]:


fns = [join("musicnet/test_data", f) for f in listdir("musicnet/test_data") if isfile(join("musicnet/test_data", f))]

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
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    # note_seq.plot_sequence(sequence_prediction)

    midi_filename = ('googlebrain/' + fn.split(".")[0].split("/")[-1] + '.mid')
    csv_filename = ('googlebrain/' + fn.split(".")[0].split("/")[-1] + '.csv')

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


# In[ ]:


import pandas as pd


# In[ ]:


df = pd.DataFrame(duration_dataset)


# In[ ]:


df.to_csv('musicnet_times.csv')


# In[ ]:




