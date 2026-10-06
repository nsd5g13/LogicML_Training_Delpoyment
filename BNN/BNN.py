# python==3.7.13, tensorflow==2.0.0

import tensorflow as tf
import larq as lq
import sys, os
import numpy as np

# --------------- load dataset ------------------------------------------------------------
dataset = sys.argv[1]
no_c_samples = int(sys.argv[2])

X_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_train_norm.npy')
X_train = np.array(X_train, dtype=np.float32)
Y_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_train.npy')
X_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_test_norm.npy')
X_test = np.array(X_test, dtype=np.float32)
Y_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_test.npy')
no_classes = len(list(set(Y_train.tolist())))

no_raw_features = len(X_train[0])
no_classes = len(list(set(Y_train)))
N_MAX = max([no_raw_features, 16, no_classes])

# -------------- model structure ---------------------------------------------------------
model = tf.keras.models.Sequential()

# In the first layer we only quantize the weights and not the input
model.add(lq.layers.QuantDense(16, use_bias=False, kernel_quantizer="ste_sign", kernel_constraint="weight_clip", input_quantizer=None))

model.add(lq.layers.QuantDense(no_classes, use_bias=False, input_quantizer="ste_sign", kernel_quantizer="ste_sign", kernel_constraint="weight_clip"))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Activation("softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------------- training and testing ----------------------------------------------------
no_epochs = 100
model.fit(X_train, Y_train, epochs=no_epochs)
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test accuracy {test_acc * 100:.2f} %")

# ------------- prediction for given samples ------------------------------------------
predictions = model.predict(X_test)
predict_labels = []
for predict in predictions:
	predict_labels.append(np.argmax(predict))
print("Actual label: %d" %Y_test[1])
print("Predicted label: %d" %predict_labels[1])

# ------------ Capture hidden layer outputs -----------------------------------------
'''
new_model = tf.keras.models.Sequential()
new_model.add(lq.layers.QuantDense(128, use_bias=False, kernel_quantizer="ste_sign", kernel_constraint="weight_clip", input_dim=160))
new_model.set_weights(model.layers[0].get_weights())
new_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
output = new_model.predict(X_test)
'''

# ------------ export test samples and labels in text files --------------------------
if not os.path.exists(r"micropython_input"):
	os.makedirs(r"micropython_input")
np.savetxt(r"micropython_input/X.txt", X_test, fmt='%d', delimiter='')
f = open(r"micropython_input/Y.txt", 'w')
f.write('predict\tactual\n')
for predict, actual in zip(predict_labels, Y_test):
	f.write('%d\t%d\n' %(predict, actual))
f.close()

# ------------ export weights in a text file --------------------------------------------
weights = model.get_weights()		# These are the full-precision weights

weights0 = weights[0]
weights0 = np.transpose(weights0)

weights0[weights0 >= 0] = 1		# 1/0 indicates +1/-1
weights0[weights0 < 0] = 0
np.savetxt(r"micropython_input/weights0.txt", weights0, fmt='%d', delimiter='')

weights1 = weights[1]
weights1 = np.transpose(weights1)
weights1[weights1 >= 0] = 1		# 1/0 indicates +1/-1
weights1[weights1 < 0] = 0
np.savetxt(r"micropython_input/weights1.txt", weights1, fmt='%d', delimiter='')

lq.models.summary(model)

# ------------- export samples/weights in c files -----------------------------------------------
if not os.path.exists(r"c_outdir"):	
	os.makedirs(r"c_outdir")

# "samples.h"
f = open(r"c_outdir/samples.h", 'w')
f.write('\n#include <stdint.h>\n\n#ifndef SAMPLES_H\n#define SAMPLES_H\n\n#define no_samples %d\n#define CLASSES %d\n#define RAW_FEATURES %d\n#define N_MAX %d\n\n' %(no_c_samples, no_classes, no_raw_features, N_MAX))
f.write('extern int32_t raw_samples[no_samples][N_MAX];\n\n#endif')
f.close()

# "samples.c"
f = open(r"c_outdir/samples.c", "w")
f.write('#include "samples.h"\n\nint32_t raw_samples[no_samples][N_MAX] = {\n')
c_samples_norm = X_test[0:no_c_samples]
c_samples = np.int32(c_samples_norm * (2**31 - 1))
for i, sample in enumerate(c_samples):
	f.write('\t{')
	for feature in sample[:-1]:
		f.write('%d, ' %int(feature))
	if i != len(c_samples)-1:
		f.write('%d},\n' %int(sample[-1]))
	else:
		f.write('%d}\n};\n' %int(sample[-1]))
f.write('\n')

p_classes_norm = model.predict(c_samples_norm)
p_classes_norm_labels = []
for predict in p_classes_norm:
	p_classes_norm_labels.append(np.argmax(predict))
p_classes = model.predict(c_samples)
p_classes_labels = []
for predict in p_classes:
	p_classes_labels.append(np.argmax(predict))
print(p_classes_norm_labels)
print(p_classes_labels)

# layer 0 
no_neurons = 16
no_inputs = len(X_train[0])

# "layer0.h"
f = open(r'c_outdir/layer0.h', 'w')
f.write('\n#include <stdint.h>\n\n#ifndef LAYER0_H\n#define LAYER0_H\n\n#define NEURONS0 %d\n#define ACTIVATIONS0 %d\n#define N_MAX %d\n\n' %(no_neurons,no_inputs, N_MAX))
f.write('extern int LAYER0[NEURONS0][N_MAX];\n\n#endif')
f.close()

# "layer0.c"
f = open(r'c_outdir/layer0.c', 'w')
f.write('#include "layer0.h"\n\nint LAYER0[NEURONS0][N_MAX] = {\n')
for each_weights in weights0[:-1]:
	f.write('\t{')
	for weight in each_weights[:-1]:
		f.write('%d, ' %weight)
	f.write('%d},\n' %each_weights[-1])
f.write('\t{')
for weight in weights0[-1][:-1]:
	f.write('%d, ' %weight)
f.write('%d}\n};' %weights0[-1][-1])
f.close()

# layer 1 
no_neurons = no_classes
no_inputs = 16

# "layer1.h"
f = open(r'c_outdir/layer1.h', 'w')
f.write('\n#include <stdint.h>\n\n#ifndef LAYER1_H\n#define LAYER0_H\n\n#define NEURONS1 %d\n#define ACTIVATIONS1 %d\n#define N_MAX %d\n\n' %(no_neurons, no_inputs, N_MAX))
f.write('extern int LAYER1[NEURONS1][N_MAX];\n\n#endif')
f.close()

# "layer1.c"
f = open(r'c_outdir/layer1.c', 'w')
f.write('#include "layer1.h"\n\nint LAYER1[NEURONS1][N_MAX] = {\n')
for each_weights in weights1[:-1]:
	f.write('\t{')
	for weight in each_weights[:-1]:
		f.write('%d, ' %weight)
	f.write('%d},\n' %each_weights[-1])
f.write('\t{')
for weight in weights1[-1][:-1]:
	f.write('%d, ' %weight)
f.write('%d}\n};' %weights1[-1][-1])
f.close()