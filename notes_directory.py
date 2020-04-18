from pydub import AudioSegment
import math

class NotesDirectory():

	def __init__(self):
		# List with the names of all available notes
		self.NOTES = [
			'A0','A#0','B0','C1','C#1',
			'D1','D#1','E1','F1','F#1',
			'G1','G#1','A1','A#1','B1',
			'C2','C#2','D2','D#2','E2',
			'F2','F#2','G2','G#2','A2',
			'A#2','B2','C3','C#3','D3',
			'D#3','E3','F3','F#3','G3',
			'G#3','A3','A#3','B3','C4',
			'C#4','D4','D#4','E4','F4',
			'F#4','G4','G#4','A4','A#4',
			'B4','C5','C#5','D5','D#5',
			'E5','F5','F#5','G5','G#5',
			'A5','A#5','B5','C6','C#6',
			'D6','D#6','E6','F6','F#6',
			'G6','G#6','A6','A#6','B6',
			'C7','C#7','D7','D#7','E7',
			'F7','F#7','G7','G#7','A7',
			'A#7','B7','C8'
		]

		# Dictionary with the frequencies of all available notes
		self.NOTE_FREQUENCIES = {
			'A0': 27.5,
			'A#0': 29.14,
			'B0': 30.87,
			'C1': 32.7,
			'C#1': 34.65,
			'D1': 36.71,
			'D#1': 38.89,
			'E1': 41.2,
			'F1': 43.65,
			'F#1': 46.25,
			'G1': 49.0,
			'G#1': 51.91,
			'A1': 55.0,
			'A#1': 58.27,
			'B1': 61.74,
			'C2': 65.41,
			'C#2': 69.3,
			'D2': 73.42,
			'D#2': 77.78,
			'E2': 82.41,
			'F2': 87.31,
			'F#2': 92.5,
			'G2': 98.0,
			'G#2': 103.83,
			'A2': 110.0,
			'A#2': 116.54,
			'B2': 123.47,
			'C3': 130.81,
			'C#3': 138.59,
			'D3': 146.83,
			'D#3': 155.56,
			'E3': 164.81,
			'F3': 174.61,
			'F#3': 185.0,
			'G3': 196.0,
			'G#3': 207.65,
			'A3': 220.0,
			'A#3': 233.08,
			'B3': 246.94,
			'C4': 261.63,
			'C#4': 277.18,
			'D4': 293.66,
			'D#4': 311.13,
			'E4': 329.63,
			'F4': 349.23,
			'F#4': 369.99,
			'G4': 392.0,
			'G#4': 415.3,
			'A4': 440.0,
			'A#4': 466.16,
			'B4': 493.88,
			'C5': 523.25,
			'C#5': 554.37,
			'D5': 587.33,
			'D#5': 622.25,
			'E5': 659.25,
			'F5': 698.46,
			'F#5': 739.99,
			'G5': 783.99,
			'G#5': 830.61,
			'A5': 880.0,
			'A#5': 932.33,
			'B5': 987.77,
			'C6': 1046.5,
			'C#6': 1108.73,
			'D6': 1174.66,
			'D#6': 1244.51,
			'E6': 1318.51,
			'F6': 1396.91,
			'F#6': 1479.98,
			'G6': 1567.98,
			'G#6': 1661.22,
			'A6': 1760.0,
			'A#6': 1864.66,
			'B6': 1975.53,
			'C7': 2093.0,
			'C#7': 2217.46,
			'D7': 2349.32,
			'D#7': 2489.02,
			'E7': 2637.02,
			'F7': 2793.83,
			'F#7': 2959.96,
			'G7': 3135.96,
			'G#7': 3322.44,
			'A7': 3520.0,
			'A#7': 3729.31,
			'B7': 3951.07,
			'C8': 4186.01
		}

		# List with audio of each note on NOTES
		self.AUDIO_NOTES = []

		for note in self.NOTES:
			self.AUDIO_NOTES.append(
				AudioSegment.from_wav(f'data/half_notes_88/{note}.wav')
			)

	def get_audio_note(self, index: int) -> AudioSegment:
		return self.AUDIO_NOTES[index]

	def get_note_frequency(self, index: int) -> float:
		return self.NOTE_FREQUENCIES[self.NOTES[index]]

	def get_closest_note(self, freq: float) -> int:
		i = 0
		min_diff, min_pos = math.inf, 0
		for key, val in self.NOTE_FREQUENCIES.items():
			diff = abs(freq - val)
			if diff < min_diff:
				min_diff = diff
				min_pos = i
			i += 1
		return min_pos

	def get_candidate_range(self, confidence: float) -> int:
		if confidence <= 0.16667:
			return 11
		elif confidence <= 0.33334:
			return 9
		elif confidence <= 0.66668:
			return 7
		elif confidence <= 0.83335:
			return 5
		else:
			return 3