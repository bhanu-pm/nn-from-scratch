import numpy as np


class Linear:
	def __init__(self, no_input_neurons, no_output_neurons, bias: bool = True):
		self.no_input_neurons = no_input_neurons
		self.no_output_neurons = no_output_neurons

		self.weights = np.random.randn(self.no_output_neurons, self.no_input_neurons) * 0.01
		self.bias_bool = bias
		if self.bias_bool:
			self.bias = np.zeros((self.no_output_neurons,))

	def __call__(self, x):
		return self.forward(x)

	def forward(self , x):
		# weights: 5x2, inputs: 2x1, + bias: 5x1, logits: 5x1
		if self.bias_bool:
			logits = (self.weights @ x) + self.bias
			return logits

		logits = self.weights @ x
		return logits
		# print(f"Output: {self.no_output_neurons}, Input: {self.no_input_neurons}")
		

class ReLU:
	def __init__(self):
		pass

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		return max(0, x)

class Softmax:
	def __init__(self):
		pass

	def __call__(self, logits: list):
		return self.forward(logits)

	def forward(self, z: list):
		stable_z = z - np.max(z)
		return np.exp(stable_z) / np.sum(np.exp(z))

if __name__ == "__main__":
	fc1 = Linear(2, 5)
	softmax = Softmax()
	inputs = [1, 2]

	logits = fc1(inputs)
	output_classes = softmax(logits)
	print(logits)
	print(output_classes)