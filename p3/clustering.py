import util
import numpy as np
import time
import csv
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import hmean
import sys
import traceback


class kMeans():
	def __init__(self):
		self.K = -1
		self.trainfile = 'train.csv'
		self.labels = []
		self.clusters = []
		self.cluster_medians = []
		self.cluster_artist_medians = []
		self.testing = False

	def train(self, k=200, trainfile='train.csv', batch=5000):
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

	def train_by_profile(self, k=200, trainfile='train.csv', batch=10000, testing=True):
		# Train the kmeans
		self.K = k
		print self.K
		self.testing = testing
		profiles = util.create_profile_matrix()
		start = time.time()
		kmeans = MiniBatchKMeans(n_clusters=self.K, batch_size=batch, compute_labels=True)
		kmeans.fit(profiles)
		print time.time() - start
		self.labels = kmeans.labels_
		self.clusters = kmeans.cluster_centers_

		# Get median number of plays per artist per cluster
		if self.testing:
			self.test, self.train = util.create_test_train(trainfile, test_amt=.2)
		else:
			self.train = util.create_big_ass_matrix(trainfile)
		self.cluster_artist_medians = [[] for i in range(self.K)]
		self.cluster_medians = [0 for i in range(self.K)]
		for i in range(self.K):
			print i
			data = self.train[np.where(self.labels == i)]
			data_masked = np.ma.masked_where(data==0, data)
			try:
				# Median per artist per cluster
				self.cluster_artist_medians[i] = [np.mean(self.reject_outliers(data_masked[:,j])) for j in range(len(data[0]))]
				# Overall median per cluster
				self.cluster_medians[i] = np.mean(self.reject_outliers(data_masked))
			except IndexError as inst:
				# If there are no points associated with a cluster, it won't be able to be
				# median-ed, print them out just to check
				print inst
				exc_type, exc_value, exc_traceback = sys.exc_info()
				print "*** print_tb:"
				traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)

		self.cluster_artist_medians = np.array(self.cluster_artist_medians)
		print self.cluster_artist_medians[np.argwhere(np.isnan(self.cluster_artist_medians))]
		print np.argwhere(np.isnan(self.cluster_artist_medians))
		print self.cluster_artist_medians.shape
		np.savetxt('kmeansclusters_profiles.csv', kmeans.cluster_centers_, delimiter=',')
		np.savetxt('kmeanslabels_profiles.csv', kmeans.labels_, delimiter=',')

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
			if not i % 1000:
				print i
			val = self.clusters[int(self.labels[int(users[row[1]])])][int(artists[row[2]])]
			val = 0 if val == 0 else np.exp(val - 1)
			writer.writerow([i, int(round(val))])
			i += 1

	def predict_profile(self, testfile='test.csv', outfile='kmeans_profiles_results.csv'):
		users = util.dict_from_csv("profiles_dict.csv")
		artists = util.dict_from_csv("artists_dict.csv")

		user_medians = [np.mean(self.reject_outliers(np.ma.masked_where(user==0, user))) for user in self.train]

		if self.testing:
			testing = self.test
		else:
			res = open(outfile, 'wb')
			writer = csv.writer(res)
			writer.writerow(['Id','plays'])
			testing = csv.reader(open(testfile, 'r'))
			next(testing, None)
		
		i = 1
		error = 0
		for row in testing:
			user = int(users[row[1]])
			cluster_artist_median = self.cluster_artist_medians[int(self.labels[user])][int(artists[row[2]])]
			cluster_median = self.cluster_medians[int(self.labels[user])]
			med = user_medians[user]
			if cluster_artist_median == 0 or cluster_median == 0:
				val = med
				if val == 0:
					val = 118
			else:
				val = (float(cluster_artist_median) / cluster_median) * med
				val = (val + med) / 2.0
			val = val * .97
			if self.testing:
				error += abs(val - int(row[3]))
			else:
				writer.writerow([i, val])
			i += 1
		if self.testing:
			error = float(error) / len(self.test)
			print "Error: {}".format(error)
			return error

	def reject_outliers(self, data, m=2):
	# 0 vals should be masked by the time you get here
		mean = np.mean(data)
		if np.ma.is_masked(mean):
			# this means it was all zeroes, to avoid nan, make it an array of 1 0 to return for mean purposes
			return [0]
		return data[abs(data - mean) < m * np.std(data)]

kmeans = kMeans()
kmeans.train_by_profile(k=10, testing=True)
kmeans.predict_profile(outfile='kmeans_profiles_results_k10_97_hmean.csv')

