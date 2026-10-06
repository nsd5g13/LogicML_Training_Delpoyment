import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time
import sys
import evaluation
import os

no_clauses = int(sys.argv[1])
T = int(sys.argv[2])
s = float(sys.argv[3])
no_epochs = int(sys.argv[4])
literal_budget = int(sys.argv[5])
dataset = sys.argv[6]

X_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_train.npy')
Y_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_train.npy')
X_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_test.npy')
Y_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_test.npy')
no_classes = len(set(Y_train))
number_of_literals = 2*len(X_train[0])
print('# %s dimensions: %d training samples, %d test samples, %d classes, %d literals' %(dataset, len(Y_train), len(Y_test), no_classes, number_of_literals))

tm = MultiClassTsetlinMachine(no_clauses, T, s, weighted_clauses=False)
tm.max_included_literals=literal_budget

all_accuracy = []
all_num_inc = []

# training
print("\nAccuracy over %d epochs:\n" %(no_epochs))
for i in range(no_epochs):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	number_of_includes = evaluation.IncludeCount(tm)

	print("#%d Test accuracy: %.2f%% Training: %.2fs Testing: %.2fs Number of Includes: %d" % (i+1, result, stop_training-start_training, stop_testing-start_testing, number_of_includes))
	all_accuracy.append(result)
	all_num_inc.append(number_of_includes)

# TM log files
if not os.path.exists(r"log"):
	os.makedirs(r"log")
evaluation.ClauseExpr(r"log/clause_expr.txt", tm) 	# clause expressions
evaluation.GetTA_action(r"log/actions.txt", tm) 	# TA actions
evaluation.TA_states(r"log/states.txt", tm) 		# TA states
evaluation.XY4verification(X_test, Y_test, tm, r"log")	# samples for verification

# accuracy and number of includes during training
result_file = open(r"log/AccuracyVsIncludes.txt", "w")
for each1, each2 in zip(all_accuracy, all_num_inc):
    result_file.write("%.2f %d;\n" %(each1, each2))
result_file.close()

# ------------------ model-dependent c codes ----------------------------------
if not os.path.exists(r"../outdir"):
	os.makedirs(r"../outdir")

# "TA_actions.h"
f = open(r'../outdir/TA_actions.h', 'w')
f.write('\n#ifndef TA_ACTIONS_H\n#define TA_ACTIONS_H\n\n#define CLASSES %d\n#define CLAUSES %d\n#define LITERALS %d\n\n' %(no_classes, no_clauses, number_of_literals))
f.write('extern int ta_actions[CLASSES*CLAUSES][LITERALS];\n\n#endif')
f.close()

# "TA_actions.c"
f = open(r'log/actions.txt', 'r')
actions = f.readlines()
f.close()
f = open(r'../outdir/TA_actions.c', 'w')
f.write('#include "TA_actions.h"\n\nint ta_actions[CLASSES*CLAUSES][LITERALS] = {\n')
for i, line in enumerate(actions):
	f.write('{')
	for action in line[:-2]:
		f.write(action + ', ')
	if i != len(actions)-1:
		f.write(line[-2] + '},\n')
	else:
		f.write(line[-2] + '}\n\n};')
f.close()

# "TA_actions_uint32.h"
f = open(r'../outdir/TA_actions_uint32.h', 'w')
no_packed_literals = int(np.ceil(number_of_literals/32))
if no_packed_literals != 1:
	f.write('\n#ifndef TA_ACTIONS_H\n#define TA_ACTIONS_H\n\n#define CLASSES %d\n#define CLAUSES %d\n#define PACKED_LITERALS %d\n\n' %(no_classes, no_clauses, no_packed_literals))
	f.write('extern int ta_actions[CLASSES*CLAUSES][PACKED_LITERALS];\n\n#endif')
else:
	f.write('\n#ifndef TA_ACTIONS_H\n#define TA_ACTIONS_H\n\n#define CLASSES %d\n#define CLAUSES %d\n\n' %(no_classes, no_clauses))
	f.write('extern int ta_actions[CLASSES*CLAUSES][1];\n\n#endif')
f.close()

# "TA_actions_uint32.c"
f = open(r'log/actions.txt', 'r')
actions = f.readlines()
f.close()
f = open(r'../outdir/TA_actions_uint32.c', 'w')
if no_packed_literals != 1:
	f.write('#include "TA_actions_uint32.h"\n\nint ta_actions[CLASSES*CLAUSES][PACKED_LITERALS] = {\n')
	for action in actions[:-1]:
		f.write('{')
		action = action.replace('\n', '')
		for i in range(no_packed_literals-1):
			packed_action = action[i*32 : (i+1)*32]
			packed_action_int = int(packed_action, 2)
			f.write('%d, ' %packed_action_int)
		packed_action = action[(no_packed_literals-1)*32:]
		packed_action_padded = packed_action.ljust(32, '0')
		packed_action_int = int(packed_action_padded, 2)
		f.write('%d},\n' %packed_action_int)
	action = actions[-1].replace('\n', '')
	f.write('{')
	for i in range(no_packed_literals-1):
		packed_action = action[i*32 : (i+1)*32]
		packed_action_int = int(packed_action, 2)
		f.write('%d, ' %packed_action_int)
	packed_action = action[(no_packed_literals-1)*32:]
	packed_action_padded = packed_action.ljust(32, '0')
	packed_action_int = int(packed_action_padded, 2)
	f.write('%d}\n};\n' %packed_action_int)
		
else:
	f.write('#include "TA_actions_uint32.h"\n\nint ta_actions[CLASSES*CLAUSES][1] = {\n')
	for action in actions[:-1]:
		action = action.replace('\n', '')
		action_padded = action.ljust(32, '0')
		action_int = int(action_padded, 2)
		f.write('{%d},\n' %action_int)
	action = actions[-1].replace('\n', '')
	action_padded = action.ljust(32, '0')
	action_int = int(action_padded, 2)
	f.write('{%d}\n\n};' %action_int)
f.close()

# "predicted_class.txt"
f = open(r'../outdir/predicted_class.txt', 'w')
c_bool_samples = np.load(r'../outdir/bool_samples.npy')
p_classes = tm.predict(c_bool_samples)
for p_class in p_classes:
	f.write(str(p_class) + '\n')
f.close()

# "predicted_class.npy"
np.save(r'../outdir/predicted_class.npy', p_classes)