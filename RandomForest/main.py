import numpy as np
import sys, random
from Forest import Forest
import os

from sklearn import datasets

dataset = sys.argv[1]
no_trees = int(sys.argv[2])
max_depth = int(sys.argv[3])
no_c_samples = int(sys.argv[4])

X_train = np.load(r'datasets/datasets/'+dataset+'/X_train.npy')
Y_train = np.load(r'datasets/datasets/'+dataset+'/Y_train.npy')
X_test = np.load(r'datasets/datasets/'+dataset+'/X_test.npy')
Y_test = np.load(r'datasets/datasets/'+dataset+'/Y_test.npy')

no_raw_features = len(X_train[0])
no_classes = len(list(set(Y_train)))

forest = Forest(max_depth=max_depth, no_trees=no_trees,
                min_samples_split=2, min_samples_leaf=1,
                feature_search=4, bootstrap=True)

X_train_new = []
Y_train_new = []
for X, Y in zip(X_train, Y_train):
	if '?' not in X:
		X_train_new.append(X.tolist())
		Y_train_new.append(Y.tolist())
X_train = np.array(X_train_new)
Y_train = np.array(Y_train_new)

X_test_new = []
Y_test_new = []
for X, Y in zip(X_test, Y_test):
	if '?' not in X:
		X_test_new.append(X.tolist())
		Y_test_new.append(Y.tolist())
X_test = np.array(X_test_new)
Y_test = np.array(Y_test_new)

print('Number of features: %d' %(len(X_train[0])))
models = forest.train(X_train[0:10000], Y_train[0:10000])

train_acc = forest.eval(X_train, Y_train)  # Retrieve train accuracy
test_acc = forest.eval(X_test, Y_test)  # Retrieve test accuracy

print('Train accuracy: %.2f%%' %(train_acc*100))
print('Test accuracy: %.2f%%' %(test_acc*100))

# For micropython -----------------------------------------------------
# save model
f = open(r'micropython_input/RF.txt', 'w')
trees = []
for i in range(no_trees):
	model = models[i]
	info = str(model).split()

	node_idx = []
	leaf = []
	for j, each in enumerate(info):
		if 'True' in each or 'False' in each:
			node_idx.append(j)
			if 'True' in each:
				leaf.append(True)
			else:
				leaf.append(False)

	tree = [[] for i in range(max_depth+1)]
	sprout_idx = [0 for i in range(max_depth+1)]
	for idx, is_leaf in zip(node_idx, leaf):
		if is_leaf == True:
			depth = int(info[idx+1])
			tree[depth].append(int(info[idx+2]))
			tree[depth].append('label')
			tree[depth].append('nan')
			tree[depth].append('nan')
		else:
			depth = int(info[idx+1])
			tree[depth].append(int(info[idx+3]))
			tree[depth].append(float(info[idx+4]))
			tree[depth].append(sprout_idx[depth])
			tree[depth].append(sprout_idx[depth]+4)
			sprout_idx[depth] = sprout_idx[depth] + 8

	tree = [x for x in tree if x != []]
	trees.append(tree)

	for each_depth in tree:
		for each in each_depth:
			f.write('%s ' %each)
		f.write('\n')

	f.write('\n')		

f.close()

# test samples
np.savetxt(r'micropython_input/X.txt', X_test, fmt='%f', delimiter=' ')
predict_Y = forest.predict(X_test[0:1000])
f = open(r"micropython_input/Y.txt", 'w')
f.write('predict\tactual\n')
for predict, actual in zip(predict_Y, Y_test):
	f.write('%d\t%d\n' %(predict, actual))
f.close()

# save result
f = open(r'log/results.txt', 'a')
f.write("%d\t%d\t%.2f\n" %(no_trees, max_depth, test_acc*100))
f.close()

# For C prog --------------------------------------------------------
if not os.path.exists(r"c_outdir"):	
	os.makedirs(r"c_outdir")

# "samples.h"
f = open(r"c_outdir/samples.h", 'w')
f.write('\n#ifndef SAMPLES_H\n#define SAMPLES_H\n\n#define no_samples %d\n#define CLASSES %d\n#define RAW_FEATURES %d\n\n' %(no_c_samples, no_classes, no_raw_features))
f.write('extern int raw_samples[no_samples][RAW_FEATURES];\n\n#endif')
f.close()

# "samples.c"
f = open(r"c_outdir/samples.c", "w")
f.write('#include "samples.h"\n\nint raw_samples[no_samples][RAW_FEATURES] = {\n')
c_samples = X_test[0:no_c_samples]
for i, sample in enumerate(c_samples):
	f.write('\t{')
	for feature in sample[:-1]:
		f.write('%d, ' %int(feature))
	if i != len(c_samples)-1:
		f.write('%d},\n' %int(sample[-1]))
	else:
		f.write('%d}\n};\n' %int(sample[-1]))
f.write('\n')

# "samples.npy"
np.save(r'c_outdir/samples.npy', c_samples)

# "predicted_class.txt"
f = open(r'c_outdir/predicted_class.txt', 'w')
p_classes = forest.predict(c_samples)
print(p_classes)
for p_class in p_classes:
	f.write(str(p_class) + '\n')
f.close()

# "predicted_class.npy"
np.save(r'c_outdir/predicted_class.npy', p_classes)

# "RF_model.h"
f = open(r'c_outdir/RF_model.h', 'w')
f.write('\n#ifndef RF_MODEL_H\n#define RF_MODEL_H\n\n#define TREES %d\n\n' %no_trees)
f.write('#include "samples.h"\n\nextern int trees[][4];\n\n#endif')
f.close()

# "RF_model.c"
f = open(r'c_outdir/RF_model.c', 'w')
f.write('#include "RF_model.h"\n\nint trees[][4] = {\n')
tree_cnt = 0
for tree in trees:
	for depth in tree:
		no_node = int(len(depth)/4)
		cnt = 0
		for j in range(no_node):
			node = depth[j*4:(j+1)*4]
			if node[1] != 'label':
				node[2] = no_node + cnt - depth.count(-1)
				node[3] = no_node + cnt + 1 - depth.count(-1)
			else:
				node[1] = -1
				node[2] = 0
				node[3] = 0
			f.write('\t{')
			for each in node[:-1]:
				f.write('%d, ' %each)
			f.write('%d},\n' %node[-1])
			cnt = cnt + 1
	if tree_cnt != len(trees)-1:
		f.write('\t{0, 0, 0, 0},\n')
		tree_cnt = tree_cnt + 1
	else:
		f.write('\t{0, 0, 0, 0}\n};')
f.close()