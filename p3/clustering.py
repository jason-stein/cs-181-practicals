import util
import numpy as np
import time
import csv
from sklearn.cluster import MiniBatchKMeans


class kMeans():
	def __init__(self):
		self.K = -1
		self.trainfile = 'train.csv'
		self.labels = []
		self.clusters = []

	def train(self, k=2000, trainfile='train.csv', batch=10000):
		self.K = k
		training = util.create_big_ass_matrix(trainfile)
		training = np.array(training)
		start = time.time()
		training_log = np.where(training==0, 0, np.log(training) + 1)
		print time.time() - start
		start = time.time()
		kmeans = MiniBatchKMeans(n_clusters=self.K, batch_size=batch, compute_labels=True)
		kmeans.fit(training_log)
		print time.time() - start
		print kmeans.cluster_centers_.shape
		self.labels = kmeans.labels_
		self.clusters = kmeans.cluster_centers_
		print kmeans.labels_.shape
		np.savetxt('kmeansclusters.csv', kmeans.cluster_centers_, delimiter=',')
		np.savetxt('kmeanslabels.csv', kmeans.labels_, delimiter=',')

	def predict(self, testfile='test.csv', outfile='kmeans_results.csv'):
		users = util.dict_from_csv("profiles_dict.csv")
		artists = util.dict_from_csv("artists_dict.csv")

		res = open(outfile, 'wb')
		writer = csv.writer(res)
		writer.writerow(['Id','plays'])

		testing = csv.reader(open(testfile, 'r'))
		next(testing, None)
		i = 1

		for row in testing:
			i += 1
			if not i % 1000:
				print i
			val = self.clusters[int(self.labels[int(users[row[1]])])][int(artists[row[2]])]
			val = 0 if val == 0 else np.exp(val - 1)
			writer.writerow([i, int(round(val))])

kmeans = kMeans()
kmeans.train()
kmeans.predict()

