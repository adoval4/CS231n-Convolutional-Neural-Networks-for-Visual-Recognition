import os
import numpy as np

def parse_data_and_labels_for_letter_class_from_string(line):
	split = line.split(',')

	label = int(split[0])
	data = []

	data.append(label)

	split = split[1:] # includes w, aspect, image
	# split = split[2:] # includes aspect, image
	# split = [split[1]] + split[3:] # includes w, image
	# split = split[3:] # includes w, image
	# split = split[4:] # includes only image

	# estimate h
	# h = float(split[1])/float(split[2]) 
	# split = [str(h)] + split[1:] # includes h, w, aspect, image
	# split = [str(h)] + split[2:] # includes h, aspect, image
	# split = [str(h)] + split[3:] # includes h, image


	for val in split:
		val = float(val)
		data.append(val)

	return data


def read_data_file(filename):
	dataset = list()
	
	if not os.path.isfile(filename) :
		print "ERROR: Data filename does not exists "+filename
		return

	with open(filename) as f:
		for line in f:
			data = parse_data_and_labels_for_letter_class_from_string(line)
			dataset.append(data)

		return dataset


def get_dataset(filename, random_order, num_samples):
	dataset = read_data_file(filename)

	if random_order:
		random.shuffle(dataset)

	if len(dataset) > num_samples >= 0:
		dataset = dataset[:num_samples]

	dataset =  np.array(dataset)	

	return dataset


def get_data_set_label_and_data(filename, num_samples, random_order = False):
	dataset = get_dataset(filename, random_order, num_samples)

	labels = []
	data = []

	for set in dataset:
		labels.append(set[0])
		data.append(set[1:])
	
	return labels, data


def get_training_and_test_sets(labels, data, training_data_proportion = 0.66):

	num_samples_for_training = int(len(data) * training_data_proportion)

	data_train, labels_train = data[:num_samples_for_training], labels[:num_samples_for_training]
	data_test, labels_test = data[num_samples_for_training:], labels[num_samples_for_training:]

	return  (np.array(data_train), np.array(labels_train)), (np.array(data_test), np.array(labels_test))