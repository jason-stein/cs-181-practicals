import csv
import numpy as np
import util
import time

artists_file = 'artists.csv'
users_file = 'profiles.csv'
train_file = 'train.csv'

artists_dict = util.dict_from_csv('artists_dict.csv')
profiles_dict = util.dict_from_csv('profiles_dict.csv')

big_ass_matrix = np.zeros((len(profiles_dict), len(artists_dict)))

start = time.time()
with open(train_file, 'r') as train:
	training = csv.reader(train, delimiter=',', quotechar='"')
	for line in training:
		if isinstance(line[2], int):
			row = profiles_dict[line[0]]
			col = artists_dict[line[1]]
			val = line[2]
			big_ass_matrix[row][col] = val

print big_ass_matrix.shape

N = len(big_ass_matrix)
M = len(big_ass_matrix[0])
K = 10

P = np.random.rand(N,K)
Q = np.random.rand(M,K)


print time.time() - start
# start = time.time()
# with open('big_ass_matrix.csv', 'wb') as BAM:
# 	writer = csv.writer(BAM)
# 	i = 0
# 	for row in big_ass_matrix:
# 		if i % 1000 == 0:
# 			print i
# 		i += 1 
# 		writer.writerow(row)
# print time.time() - start

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
    	start = time.time()
    	print step
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
     	print time.time() - start
    return P, Q.T
start = time.time()
nP, nQ = matrix_factorization(big_ass_matrix, P, Q, K)
nR = np.dot(nP, nQ.T)
print time.time() - start