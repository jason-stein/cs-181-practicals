import csv
import numpy as np
import util
import time
from sklearn.decomposition import NMF
from scipy import sparse


big_ass_matrix = np.load('train_matrix.npy')
sBAM = sparse.csr_matrix(big_ass_matrix)

print "Matrix loaded."
print ""

start = time.time()
model = NMF(n_components=10, tol=.01)
model.inverse_transform(model.fit_transform(sBAM))
print "5 iters: " + str(time.time() - start)
print mat

# for i in xrange(sBAM.shape[0]):
# 	for j in xrange(sBAM.shape[1]):
# 		if sBAM[i,j] > 0:
# 			print i, j