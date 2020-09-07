import time
import librosa
import sys
import vamp

import pandas as pd

from os import listdir
from os.path import isfile, join

def transcribe_audio(data_path="musicnet/test_data", transcription_path="data/transcription/silvet/"):
    duration_dataset = {
        'song': [],
        'inference_time_in_seconds': []
    }

    fns = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
    for fn in fns:
        print(fn)
        if fn == "musicnet/test_data/2191.wav":
            continue

        now = time.time()

        data, rate = librosa.load(fn)
        transcription = vamp.collect(data, rate, "silvet:silvet")

        now_now = time.time()

        song_df_dict = {
            'onset_time': [],
            'offset_time': [],
            'pitch': [],
            'note': []
        }
        for n in transcription['list']:
            song_df_dict['onset_time'].append(float(n["timestamp"]) * 1000)
            song_df_dict['offset_time'].append((float(n["timestamp"]) * 1000) + (float(n["duration"]) * 1000))
            song_df_dict['pitch'].append(n["values"][0])
            song_df_dict['note'].append(n["values"][1])
        
        csv_filename = (transcription_path + fn.split(".")[0].split("/")[-1] + '.csv')
        song_df = pd.DataFrame(song_df_dict)
        song_df.to_csv(csv_filename)
        
        duration_dataset['song'].append(fn.split("/")[-1])
        duration_dataset['inference_time_in_seconds'].append(now_now - now)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        transcribe_audio(sys.argv[1], sys.argv[2])
    else:
        transcribe_audio()
