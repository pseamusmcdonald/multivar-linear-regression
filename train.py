from sklearn import preprocessing
import csv
import numpy as np

CONVERGENCE_THRESH = .1
ALPHA = .1

## Mean Squared Error (Hypo vs Truths)

def cost_function(thetas):
	m = len(train_normalized)
	sqError = (1/(2*m))*(np.sum(np.square(np.subtract(np.matmul(train_normalized, thetas), truths_raw))))
	return sqError

with open('data/truths.csv', 'r') as truths_file, open('data/training_inputs.csv') as train_file:
	truths_data = csv.reader(truths_file, delimiter = ',')
	train_data = csv.reader(train_file, delimiter = ',')
	truths_temp = [data for data in truths_data]
	train_temp = [data for data in train_data]


train_raw = np.array(train_temp, dtype=float)
truths_raw = np.array(truths_temp, dtype=float)

train_added = np.concatenate((np.ones((len(train_raw), 1)), train_raw), axis=1)

## Normalize Data

train_raw_transposed = np.transpose(train_added)
truths_raw_transposed = np.transpose(truths_raw)

## Adding Constant Column

for i, val in enumerate(train_raw_transposed):
	train_raw_transposed[i] = preprocessing.normalize(train_raw_transposed[i].reshape(1, len(train_raw_transposed[i])))

for i, val in enumerate(truths_raw_transposed):
	truths_raw_transposed[i] = preprocessing.normalize(truths_raw_transposed[i].reshape(1, len(truths_raw_transposed[i])))

train_normalized = np.transpose(train_raw_transposed)
truths_normalized = np.transpose(truths_raw_transposed)

## Initialize Parameters

thetas = np.ones(len(train_normalized[0]))

last_cost = 0
number_of_steps = 0

truths_raw = np.array(truths_temp, dtype=float)

print(train_normalized)

while (cost_function(thetas) != last_cost):
	last_cost = cost_function(thetas)
	theta_temps = []
	for i, theta in enumerate(thetas):
		hypoResMatrix = np.matmul(train_normalized, thetas)
		subtractedResMatrix = np.subtract(hypoResMatrix, truths_raw)
		thetaMulResMatrix = np.matmul(subtractedResMatrix, [val[i] for val in train_normalized])
		theta_temps.append(theta - (ALPHA*((1/len(train_normalized))*np.sum(thetaMulResMatrix))))
	thetas = theta_temps
	number_of_steps += 1
print(thetas, number_of_steps)