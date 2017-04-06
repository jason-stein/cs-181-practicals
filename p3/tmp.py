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
		self.testing = False

	def train(self, k=2000, trainfile='train.csv', batch=10000, testing=True):
		self.K = k
		self.testing = testing
		if self.testing:
			self.test, self.train = util.create_test_train(trainfile, test_amt=.2)
		else:
			self.train = util.create_big_ass_matrix(trainfile)
		drop_zeroes = np.where(self.train == 0, np.nan, self.train)
		self.user_medians = [np.nanmedian(user) for user in drop_zeroes]
		start = time.time()
		training_log = np.where(self.train==0, 0, np.log(self.train) + 1)
		print time.time() - start
		start = time.time()
		kmeans = MiniBatchKMeans(n_clusters=self.K, batch_size=batch, compute_labels=True)
		kmeans.fit(training_log)
		print time.time() - start
		print kmeans.cluster_centers_.shape
		self.labels = kmeans.labels_
		self.clusters = kmeans.cluster_centers_
		self.overall_averages = [np.mean(cluster) for cluster in self.clusters]
		print kmeans.labels_.shape
		np.savetxt('kmeansclusters.csv', kmeans.cluster_centers_, delimiter=',')
		np.savetxt('kmeanslabels.csv', kmeans.labels_, delimiter=',')

	def predict(self, testfile='test.csv', outfile='kmeans_results.csv'):
		users = util.dict_from_csv("profiles_dict.csv")
		artists = util.dict_from_csv("artists_dict.csv")

		res = open(outfile, 'wb')
		writer = csv.writer(res)
		writer.writerow(['Id','plays'])

		if self.testing:
			testing = self.test
		else:
			testing = csv.reader(open(testfile, 'r'))
			next(testing, None)
		print "Predicting"
		error = 0
		for row in testing:
			# if not row[0] % 1000:
			# 	print row[0]
			user = int(users[row[1]])
			label = int(self.labels[user])
			artist = int(artists[row[2]])
			val = self.clusters[label][artist]
			val = 0 if val == 0 else np.exp(val - 1)
			val -= self.overall_averages[label]
			val += self.user_medians[user]
			if self.testing:
				error += abs(val- int(row[3]))
			else:
				writer.writerow([row[0], int(round(val))])
		if self.testing:
			error = float(error) / len(self.test)
			print "Error: {}".format(error)
			return error

kmeans = kMeans()
kmeans.train(k=200)
kmeans.predict()

