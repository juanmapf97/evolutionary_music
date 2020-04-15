import random
import numpy as np
from scipy._lib._util import check_random_state

from jmetal.core.operator import Mutation
from jmetal.core.solution import BinarySolution

class UniformMutation(Mutation[BinarySolution]):

	def __init__(self, probability: float):
		super(UniformMutation, self).__init__(probability = probability)

	def execute(self, solution: BinarySolution) -> BinarySolution:
		offspring = solution.variables
		popsize = len(solution.variables[0])

		values = [np.random.choice(solution.variables[1][i]) for i in range(popsize)]
		# values = [np.random.choice(solution.variables[2]) for i in range(popsize)]
		probs = np.random.uniform(0, 1, popsize)
		rng = check_random_state(None)
		mask = rng.choice(
			[True, False],
			p = [self.probability, 1 - self.probability],
			size = popsize
		)
		# for i in range(popsize):
		# 	if probs[i] <= self.probability:
		# 		offspring[0][i] = values[i]
		# offspring[0][mask] = values
		for i in range(popsize):
			if mask[i]:
				offspring[0][i] = values[i]
		
		solution.variables = offspring
		return solution

	def get_name(self):
		return 'Uniform mutation'
