import numpy as np


np.random.seed(42)

class Module:
	def __init__(self):
		pass

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		return x

	def to(self, device: str):
		if device.lower() == "cpu":
			# Code to run it normally
			pass

		elif (device.lower() == "cuda") or (device.lower() == "cuda:0"):
			# Code to send it to GPU
			pass


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
		

class Dropout:
	def __init__(self, probability):
		# dropout probability * 100, % of the neurons
		self.probability = probability
		self.scaling_val = 1/(1 - self.probability)

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		mask = (np.random.uniform(0, 1, *x.shape) > self.probability).astype(float)
		dropped_x = mask * x
		scaled_dropped_x = self.scaling_val * dropped_x
		return scaled_dropped_x


class ReLU:
	def __init__(self):
		pass

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		zero = np.zeros_like(x)
		return np.maximum(0, x)

class Softmax:
	def __init__(self):
		pass

	def __call__(self, logits: list):
		return self.forward(logits)

	def forward(self, z: list):
		stable_z = z - np.max(z)
		return np.exp(stable_z) / np.sum(np.exp(z))



if __name__ == "__main__": #######################################################################################################

	class ANN(Module):
		def __init__(self):
			super(ANN, self).__init__()
			self.fc1 = Linear(8, 32)
			self.fc2 = Linear(32, 64)
			self.out = Linear(64, 10)

			self.dropout = Dropout(0.5)
			self.relu = ReLU()
			self.softmax = Softmax()

		def forward(self, x):
			x = self.fc1(x)
			print(f"Logits after fc1: {x}")
			print(f"shape: {x.shape}")

			x = self.dropout(x) 
			print(f"Logits after fc1 -> dropout: {x}")
        
			x = self.relu(x)
			print(f"Logits after fc1 -> dropout -> relu: {x}")
	
			x = self.fc2(x)
			print(f"Logits after fc2: {x}")

			x = self.dropout(x)
			print(f"Logits after fc2 -> dropout: {x}")

			x = self.relu(x)
			print(f"Logits after fc2 -> dropout -> relu: {x}")

			logits = self.out(x)
			print(f"Logits after out: {logits}")

			output_classes = self.softmax(logits)
			print(f"Outputs: {output_classes}")
        	
			return output_classes

	inputs = np.linspace(0, 8, 8)
	print(inputs)

	model = ANN()
	final_outputs = model(inputs)