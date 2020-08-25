import librosa

class OnsetDetection:
    def get_onset_times(self, wav_file_path):
        x, sr = librosa.load(wav_file_path)
        onset_frames = librosa.onset.onset_detect(x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
        onset_times = librosa.frames_to_time(onset_frames)
        return onset_times
    
    def get_audio_duration(self, wav_file_path):
        x, sr = librosa.load(wav_file_path)
        audio_duration = librosa.get_duration(x, sr)
        return audio_duration

    def get_durations(self, wav_file_path):
        onset_times = self.get_onset_times(wav_file_path)
        audio_duration = self.get_audio_Duration(wav_file_path)
        onset_times.append(audio_duration)

        durations = []
        for i in range(onset_times-1):
            durations.append(onset_times[i+1] - onset_times[i])
        
        return durations

    def get_number_of_notes(self, wav_file_path):
        x, sr = librosa.load(wav_file_path)
        onset_frames = librosa.onset.onset_detect(x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
        print(wav_file_path, len(onset_frames))
        return len(onset_frames)