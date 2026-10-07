# python==3.12.6

import os, sys, random
import numpy as np
import preprocessing

from sklearn import datasets
from keras.datasets import mnist, fashion_mnist, cifar10

# dataset name as the argument: 'emg', 'iris', 'sports', 'har', 'digits', 'gesture', 'gas', 'statlog', 'mammography', 'sensorless', 'mnist', 'kmnist', 'fmnist', 'cifar', 'kws'
all_datasets = [sys.argv[1].lower()]

# make directory to store the booleanized dataset(s)
if not os.path.exists(r"bool_datasets"):
	os.makedirs(r"bool_datasets")

# number of test samples for TM c program
no_c_samples = int(sys.argv[2])

for dataset in all_datasets:
	match dataset:
		# ------------- MNIST --------------------------------------------------
		case "mnist":
			(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
			#dist_max_min = np.max(X_train) - np.min(X_train)
			#X_train_norm = X_train / (dist_max_min/2) - 1
			#X_test_norm = X_test / (dist_max_min/2) - 1
			q_low = np.percentile(X_train, 1, axis=0)
			q_high = np.percentile(X_train, 99, axis=0)
			X_train_clip = np.clip(X_train, q_low, q_high)
			X_test_clip = np.clip(X_test, q_low, q_high)
			X_train_norm = (2 * (X_train_clip - q_low) / (q_high - q_low) - 1)
			X_test_norm = (2 * (X_test_clip - q_low) / (q_high - q_low) - 1)
			np.save(r"bool_datasets/"+dataset+'/X_train_norm.npy', X_train_norm.reshape((X_train.shape[0], 28*28)))
			np.save(r"bool_datasets/"+dataset+'/X_test_norm.npy', X_test_norm.reshape((X_test.shape[0], 28*28)))
			no_raw_features = 28*28
			c_samples = X_test.reshape((X_test.shape[0], 28*28))[0:no_c_samples]
			X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
			X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)
			c_bool_samples = X_test[0:no_c_samples]
			no_bool_bits = 1
			thresholds = [[75] for i in range(no_raw_features)]			

		# ------------- CIFAR --------------------------------------------------
		case "cifar":
			(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
			Y_train=Y_train.reshape(Y_train.shape[0])
			Y_test=Y_test.reshape(Y_test.shape[0])
			animals = np.array([2, 3, 4, 5, 6, 7])
			Y_train = np.where(np.isin(Y_train, animals), 1, 0)
			Y_test = np.where(np.isin(Y_test, animals), 1, 0)
			X_raw_features = np.concatenate((X_train_org, X_test_org), axis=0)
			X, X_hog, thresholds = preprocessing.CIFAR_HOG(X_raw_features)

			no_raw_features = len(X_hog[0])
			no_bool_bits = 1

			X_train = X[0:len(Y_train)]
			X_test = X[-len(Y_test):]

			X_train_hog = X_hog[0:len(Y_train)]
			X_test_hog = X_hog[-len(Y_test):]
			q_low = np.percentile(X_train_hog, 1, axis=0)
			q_high = np.percentile(X_train_hog, 99, axis=0)
			X_train_clip = np.clip(X_train_hog, q_low, q_high)
			X_test_clip = np.clip(X_test_hog, q_low, q_high)
			X_train_norm = (2 * (X_train_clip - q_low) / (q_high - q_low) - 1)
			X_test_norm = (2 * (X_test_clip - q_low) / (q_high - q_low) - 1)
			np.save(r"bool_datasets/"+dataset+'/X_train_norm.npy', X_train_norm)
			np.save(r"bool_datasets/"+dataset+'/X_test_norm.npy', X_test_norm)			

			c_samples = np.round(np.array(X_test_hog[:no_c_samples]) * 100000).astype(int)
			thresholds =  [[int(np.round(float(x*100000)))] for x in thresholds]
			c_bool_samples = X_test[0:no_c_samples]

			X_train=np.array(X_train)
			X_test=np.array(X_test)

		# ------------- IRIS ----------------------------------------------------
		case "iris":
			iris = datasets.load_iris()
			no_raw_features = len(iris.data[0])
			X, thresholds = preprocessing.onehot_encoding(iris.data, 3)
			thresholds = [[int(np.round(float(val*10))) for val in pair] for pair in thresholds]
			Y = iris.target
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)
			X_train_raw, _, X_test_raw, _ = preprocessing.DatasetSplit(iris.data, Y, 0.8)
			#dist_max_min = np.max(X_train_raw) - np.min(X_train_raw)
			#X_train_norm = X_train_raw / (dist_max_min/2) - 1
			#X_test_norm = X_test_raw / (dist_max_min/2) - 1
			q_low = np.percentile(X_train_raw, 1, axis=0)
			q_high = np.percentile(X_train_raw, 99, axis=0)
			X_train_clip = np.clip(X_train_raw, q_low, q_high)
			X_test_clip = np.clip(X_test_raw, q_low, q_high)
			X_train_norm = (2 * (X_train_clip - q_low) / (q_high - q_low) - 1)
			X_test_norm = (2 * (X_test_clip - q_low) / (q_high - q_low) - 1)
			np.save(r"bool_datasets/"+dataset+'/X_train_norm.npy', X_train_norm)
			np.save(r"bool_datasets/"+dataset+'/X_test_norm.npy', X_test_norm)
			c_samples = X_test_raw[0:no_c_samples]*10
			c_bool_samples = X_test[0:no_c_samples]
			no_bool_bits = 3

		# ------------- KMNIST --------------------------------------------------
		case "kmnist":
			X_train = np.load(r"raw_dataset/kmnist/kmnist-train-imgs.npz")['arr_0']
			X_test = np.load(r"raw_dataset/kmnist/kmnist-test-imgs.npz")['arr_0']
			Y_train = np.load(r"raw_dataset/kmnist/kmnist-train-labels.npz")['arr_0']
			Y_test = np.load(r"raw_dataset/kmnist/kmnist-test-labels.npz")['arr_0']
			X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
			X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

		# ------------- FMNIST --------------------------------------------------
		case "fmnist":
			(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
			X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
			X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

		# ------------- Keyword Spotting (CUDA support must be explicitly allocated) ----------------------------------------
		case "kws":
			[train_x, Y_train, test_x, Y_test] = preprocessing.kws_dataset(r"raw_dataset/mini_speech_commands")
			X_raw_features = np.concatenate((train_x, test_x), axis=0)
			X = preprocessing.thermo_encoding(X_raw_features, 3)

			X_train = X[0:len(Y_train)]
			X_test = X[-len(Y_test):]

			X_train=np.array(X_train)
			X_test=np.array(X_test)								

		# ------------- Digits ----------------------------------------------------
		case "digits":
			digits = datasets.load_digits()
			X = np.where(digits.data > 7.5, 1, 0)
			Y = digits.target
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)		

		# --------------- EMG ------------------------------------------------
		case "emg":
			PoolingWindow = 100
			PoolingMethod = 'rms'
			SlidingWinLen = 20
			Stride = 3

			print('Loading EMG dataset from given path:')
			X, Y = preprocessing.EMG_load(r'raw_dataset/EMG')
			X, Y = preprocessing.pooling(X, Y, PoolingWindow, PoolingMethod)
			X, Y = preprocessing.SlidingWindow(X, Y, SlidingWinLen, Stride)
			
			# exclude samples of class 0 from quantile binning process, as class 0 indicates unmarked data
			X_notClass0 = X[Y != 0]
			Y_notClass0 = Y[Y != 0]
			X_Class0 = X[Y == 0]
								
			X_notClass0, thresholds = preprocessing.onehot_encoding(X_notClass0, 2)
			
			# relabel unmarked data
			print('Relabelling unmarked data ... ')
			thresholds = sum(thresholds, [])
			X_Class0 = (X_Class0 > thresholds).astype(int)
			Y_relabelled = preprocessing.EMG_relabel(X_Class0, X_notClass0, Y_notClass0)
			X = np.concatenate((X_Class0, X_notClass0), axis=0)
			Y = np.concatenate((Y_relabelled, Y_notClass0), axis=0)
			
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Sports ------------------------------------------------
		case "sports":
			PoolingWindow = 10
			PoolingMethod = 'avg'
			SlidingWinLen = 1
			Stride = 1

			print('Loading Sports dataset from given path:')
			X, Y = preprocessing.Sports_load(r'raw_dataset/Sports')
			X, Y = preprocessing.pooling(X, Y, PoolingWindow, PoolingMethod)
			X, Y = preprocessing.SlidingWindow(X, Y, SlidingWinLen, Stride)

			X, _ = preprocessing.onehot_encoding(X, 2)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)
	
		# --------------- Human acitivity ------------------------------------------------
		case "har":
			X_train, Y_train = preprocessing.HAR_load(r'raw_dataset/HAR/train')
			X_test, Y_test = preprocessing.HAR_load(r'raw_dataset/HAR/test')

			X_train = np.array(X_train)
			X_test = np.array(X_test)
			X = np.concatenate((X_train, X_test), axis=0)
			
			X, _ = preprocessing.onehot_encoding(X, 2)
			X_train = X[:len(Y_train)]
			X_test = X[len(Y_train):]

		# --------------- Gesture phase segmentation --------------------------------
		case "gesture":
			X, Y = preprocessing.Gesture_load(r'raw_dataset/Gesture')
			X = np.array(X)
			Y = np.array(Y)
			X, _ = preprocessing.onehot_encoding(X, 10)
			#X = preprocessing.thermo_encoding(X, 20)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Gas Sensor Array Drift ----------------------------------------
		case "gas":
			X, Y = preprocessing.Gas_load(r'raw_dataset/Gas')
			X = np.array(X)
			Y = np.array(Y)
			X, _ = preprocessing.onehot_encoding(X, 2)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Statlog Vehicle Silhouettes ----------------------------------------
		case "statlog":
			X, Y = preprocessing.Statlog_load(r'raw_dataset/Statlog')
			X = np.array(X)
			Y = np.array(Y)
			X = preprocessing.thermo_encoding(X, 20)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Mammographic Mass ------------------------------------------------
		case "mammography":
			X, Y = preprocessing.Mammography_load(r'raw_dataset/Mammography/mammographic_masses.data')
			X = preprocessing.Mammography_encoding(X, 3, 'onehot')
			X = np.array(X)
			Y = np.array(Y)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Sensorless Drive Diagnosis ------------------------------------------------
		case "sensorless":
			X, Y = preprocessing.Sensorless_load(r'raw_dataset/Sensorless/Sensorless_drive_diagnosis.txt')
			X = np.array(X)
			Y = np.array(Y)
			X, _ = preprocessing.onehot_encoding(X, 3)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)						
														
		case _:
			print("The given dataset %s is not recognized." %sys.argv[1])

	# store datasets in given directory
	if not os.path.exists(r"bool_datasets/"+dataset):
		os.makedirs(r"bool_datasets/"+dataset)
	np.save(r"bool_datasets/"+dataset+'/X_train.npy', X_train)
	#np.savetxt(r"bool_datasets/"+dataset+'/X_train.txt', X_train, fmt='%d')
	np.save(r"bool_datasets/"+dataset+'/Y_train.npy', Y_train)
	#np.savetxt(r"bool_datasets/"+dataset+'/Y_train.txt', Y_train, fmt='%d')
	np.save(r"bool_datasets/"+dataset+'/X_test.npy', X_test)
	np.save(r"bool_datasets/"+dataset+'/Y_test.npy', Y_test)

# ------------------ dataset-dependent c codes ----------------------------------
if not os.path.exists(r"../outdir"):
	os.makedirs(r"../outdir")

# "samples.h"
f = open(r"../outdir/samples.h", 'w')
f.write('\n#ifndef SAMPLES_H\n#define SAMPLES_H\n\n#define no_samples %d\n#define RAW_FEATURES %d\n#define no_bits %d\n\n' %(no_c_samples, no_raw_features, no_bool_bits))
if no_bool_bits != 1:
	f.write('extern int raw_samples[no_samples][RAW_FEATURES];\nextern int thresholds[RAW_FEATURES][no_bits-1];\n\n#endif')
else:
	f.write('extern int raw_samples[no_samples][RAW_FEATURES];\nextern int thresholds[RAW_FEATURES][1];\n\n#endif')
f.close()

# "samples.c"
f = open(r"../outdir/samples.c", "w")
f.write('#include "samples.h"\n\nint raw_samples[no_samples][RAW_FEATURES] = {\n')
for i, sample in enumerate(c_samples):
	f.write('\t{')
	for feature in sample[:-1]:
		f.write('%d, ' %int(feature))
	if i != len(c_samples)-1:
		f.write('%d},\n' %int(sample[-1]))
	else:
		f.write('%d}\n};\n' %int(sample[-1]))
f.write('\n')
if no_bool_bits != 1:
	f.write('int thresholds[RAW_FEATURES][no_bits-1] = {\n')
else:
	f.write('int thresholds[RAW_FEATURES][1] = {\n')
for i, threshold in enumerate(thresholds):
	if no_bool_bits == 1:
		if i != no_raw_features-1:
			f.write('\t{%d},\n' %threshold[0])
		else:
			f.write('\t{%d}\n};' %threshold[0])
	else:
		f.write('\t{')
		for each in threshold[:-1]:
			f.write('%d, ' %int(each))
		if i != no_raw_features-1:
			f.write('%d},\n' %int(threshold[-1]))
		else:
			f.write('%d}\n};\n' %int(threshold[-1]))
f.close()

# "samples.npy"
np.save(r'../outdir/samples.npy', c_samples)

# "bool_samples.txt"
f = open(r"../outdir/bool_samples.txt", "w")
for sample in c_bool_samples:
	for feature in sample[:-1]:
		f.write('%d ' %int(feature))
	f.write('%d\n' %int(sample[-1]))
f.close()

# "bool_samples.npy"
np.save(r'../outdir/bool_samples.npy', c_bool_samples)



