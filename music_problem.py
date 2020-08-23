import numpy as np
from pydub import AudioSegment
from notes_directory import NotesDirectory
from audio_comparer import AudioComparer

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution

from onset_detection import OnsetDetection

# Analizing frequencies
from scipy.io import wavfile
import crepe

class MusicProblem(BinaryProblem):

	def __init__(self, target_name: str, number_of_notes: int):
		self.number_of_notes = number_of_notes
		self.number_of_variables = 3 # Change to add time interval fitness
		self.number_of_objectives = 1
		self.number_of_constraints = 0
		self.obj_labels = ['similarity']
		self.obj_directions = [self.MAXIMIZE]
		self.audio_comparer = AudioComparer(target_name)
		self.notes_directory = NotesDirectory()
		self.number_of_available_notes = len(self.notes_directory.NOTES)

		self._onsets = []
		self._durations = []
		self.note_candidates = self.generate_candidates()

	def generate_candidates(self):
		sr, audio = wavfile.read(self.audio_comparer.target_name)

		self._onsets = OnsetDetection.get_onset_times(self.audio_comparer.target_name)
		self._durations = OnsetDetection.get_durations(self.audio_comparer.target_name)

		time, frequency, confidence, activation = crepe.predict(audio, sr, step_size=350)
		
		candidates = []
		for i in range(self.number_of_notes):
			if i < len(frequency):
				closest_note_position = self.notes_directory.get_closest_note(frequency[i])
				range_num = self.notes_directory.get_candidate_range(confidence[i])
				
				candidate_notes = list(range(max(closest_note_position - range_num // 2, 0), min(closest_note_position + range_num // 2 + 1, self.number_of_available_notes)))
				candidates.append(candidate_notes)
			else:
				candidates.append(list(range(self.number_of_available_notes)))
		return candidates

	def evaluate(self, solution: BinarySolution) -> BinarySolution:
		combined = AudioSegment.empty()
		for note_index in solution.variables[0]:
			combined += self.notes_directory.get_audio_note(note_index)
		combined.export('data/conc.wav', format='wav')

		solution.objectives[0] = \
			self.audio_comparer.compare('data/conc.wav')

		return solution

	def create_solution(self) -> BinarySolution:
		solution = BinarySolution(
			self.number_of_variables,
			self.number_of_objectives
		)

		# Choose notes from candidates.
		notes = [np.random.choice(self.note_candidates[i]) for i in range(self.number_of_notes)]
		
		solution.variables[0] = notes
		solution.variables[1] = self.note_candidates
		solution.variables[2] = self.number_of_available_notes
		solution.variables[3] = self._onsets
		solution.variables[4] = self._durations

		return solution

	def get_name(self):
		return 'Music problem'
