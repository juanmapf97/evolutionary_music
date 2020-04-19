import os, glob, wave, csv
from sklearn.metrics import mean_squared_error
from notes_directory import NotesDirectory

from pydub import AudioSegment
from scipy.io import wavfile as wav
import pandas as pd
import numpy as np

path = 'data/full_notes_88'
new_path = 'data/half_notes_88'

def process_notes():
    global path, new_path
    for filename in glob.glob(os.path.join(path, '*.wav')):
        audio = wave.open(filename, 'r')
        framerate = audio.getframerate()
        channels = audio.getnchannels()
        samp = audio.getsampwidth()

        chunk = audio.readframes(int(0.350 * framerate))

        new = wave.open(filename.replace("full_notes_88", "half_notes_88"), 'w')
        new.setnchannels(channels)
        new.setsampwidth(samp)
        new.setframerate(framerate)
        new.writeframes(chunk)
        new.close()
        audio.close()

def print_notes():
    global path, new_path
    appended = []
    for filename in glob.glob(os.path.join(path, '*.wav')):
        sections = filename.split('/')
        note = sections[2].split('.')[0]
        appended.append(f"'{note}'")
    appended.sort()
    print(','.join(appended))

def generate_frequency_dictionary():
    with open('note_frequencies.csv', 'r') as infile:
        reader = csv.reader(infile)
        freq_dict = {row[0]:float(row[1]) for row in reader}
        return freq_dict

def generate_results_csv():
    rate, target_data = wav.read('data/songs/ode_to_joy.wav')
    notes_directory = NotesDirectory()
    file_name = 'results_ode_to_joy_candidates'
    song = 'ode_to_joy'
    num = 16
    # Solution(variables=[[27, 27, 27, 27, 27, 3, 27, 11, 3], [[27], [19, 27, 34], [27], [19, 27, 34], [27], [3, 11, 19, 27, 34], [19, 27, 34], [3, 11, 19, 27, 34], [3, 11, 19, 34]]],objectives=[-0.016101511818249942],constraints=[])
    with open(f'results/{file_name}.txt', 'r') as f:
        with open(f'results_csv/{file_name}.csv', 'a') as c:
            c.write('song,number of notes,result variables,objective value,mse with target\n')
            for line in f.readlines():
                if (len(line) <= 1):
                    continue
                variables = line[line.find('[[') + 2:line.find(']')].split(', ')
                variables = list(map(int, variables))
                combined = AudioSegment.empty()
                for note_index in variables:
                    combined += notes_directory.get_audio_note(note_index)
                combined.export('data/conc.wav', format='wav')
                rate, data = wav.read('data/conc.wav')
                # 15434
                mse = mean_squared_error(target_data[:,1], data[:,1])
                objective = line[line.find('objectives=[') + 12:line.find('],constraints')]
                c.write(f'{song},{num},{variables},{objective},{mse}\n')
            
def generate_results_csv_fft():
    file_name = 'fft_no_candidates_jingle2'
    df = pd.read_csv(f'results_csv/{file_name}.csv')
    
    with open('results_csv/test.csv', 'w') as f:
        rate, target_data = wav.read('data/songs/jingle2.wav')
        notes_directory = NotesDirectory()
        for index, row in df.iterrows():
            combined = AudioSegment.empty()
            for note_index in range(11):
                combined += notes_directory.get_audio_note(int(row[f'{note_index}']))
            combined.export('data/conc.wav', format='wav')
            rate, data = wav.read('data/conc.wav')
            mse = mean_squared_error(target_data[:,1], data[:,1])
            f.write(f'{mse}\n')

    
def export_song_data():
    notes_directory = NotesDirectory()
    # rate, data = wav.read('data/songs/elise2.wav')
    combined = AudioSegment.empty()
    notes = [27, 19, 27, 6, 27, 19, 27, 5, 8]
    for note_index in notes:
        combined += notes_directory.get_audio_note(note_index)
    combined.export('data/conc.wav', format='wav')
    # rate2, data2 = wav.read('data/conc.wav')
    # c = np.column_stack((data[:,1].T, data2[:,1].T))
    # print(np.shape(c))
    # np.savetxt('exported/comp.txt', c, delimiter=',')
    # with open('exported/elise2.txt', 'w') as ex:
    #     ex.write(data[:,1])



# test = generate_frequency_dictionary()
# print(test)
# generate_results_csv()
# generate_results_csv_fft()
export_song_data()