import sys

from pydub import AudioSegment
from music_problem import MusicProblem
from notes_directory import NotesDirectory

from jmetal.operator import SPXCrossover, BinaryTournamentSelection, BitFlipMutation
from jmetal.util.observer import ProgressBarObserver, PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm

from uniform_crossover import UniformCrossover
from uniform_mutation import UniformMutation

from onset_detection import get_number_of_notes

def generate_audio_file(notes, dest):
	notes_directory = NotesDirectory()

	combined = AudioSegment.empty()
	for var in notes:
		combined += notes_directory.get_audio_note(var)
	combined.export(dest, format = 'wav')

# Swan - 20
# Hear & Soul - 18
# twinkle - 28
def evo_music(source='data/songs/elise2.wav', dest='data/generated_songs/elise2.wav'):
	number_of_notes = get_number_of_notes(source)
	problem = MusicProblem(source, number_of_notes)
	population_size = 100
	max_evaluations = 1500

	algorithm = GeneticAlgorithm(
		problem = problem,
		mutation = UniformMutation(probability = 1.0 / problem.number_of_notes),
		crossover = UniformCrossover(1.0),
		selection = BinaryTournamentSelection(),
		population_size = population_size,
		termination_criterion = StoppingByEvaluations(max_evaluations=max_evaluations),
		offspring_population_size = population_size
	)

	# Initialize progress bar observer
	algorithm.observable.register(observer = PrintObjectivesObserver())

	# for i in range(30):
	# Run algorithm and set results
	algorithm.run()
	front = algorithm.get_result()

	# Create audio file from the best result
	generate_audio_file(front.variables[0], dest)

	return front

if __name__ == "__main__":
 	evo_music()