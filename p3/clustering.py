import util
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans


def kmeans(k=200):
	training = util.create_big_ass_matrix('train.csv')
	training = np.array(training)
	start = time.time()
	training_log = np.where(training==0, 0, np.log(training) + 1)
	print time.time() - start
	start = time.time()
	kmeans = MiniBatchKMeans(n_clusters=1000, batch_size=1000)
	kmeans.fit(training_log)
	print time.time() - start
	print kmeans.cluster_centers_.shape
	testing = util.create_big_ass_matrix('test.csv')
	testing = np.array(testing)
	testing_log = np.where(testing==0,0, np.log(testing)+1)
	results = kmeans.predict(testing_log)
	np.savetxt('kmeansres.csv', results, delimiter=',')
	np.savetxt('kmeansclusters.csv', kmeans.cluster_centers_, delimiter=',')

kmeans()

