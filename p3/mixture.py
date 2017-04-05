import util
import numpy as np
import time
import csv
from sklearn.mixture import GaussianMixture

class gMixture:
	def __init__(self):
		self.n_components = -1
		self.trainfile = 'train.csv'

	def fit(self, n_components = 200, trainfile = 'train.csv'):
		self.n_components = n_components
		training = util.create_big_ass_matrix(trainfile)
		training = np.array(training)
		start = time.time()
		training_log = np.where(training==0, 0, np.log(training) + 1)
		print "Time to find logs: " + str(time.time() - start)
		start = time.time()
		model = GaussianMixture(n_components = self.n_components, max_iter = 10)
		print "Fitting"
		model.fit(training_log)
		self.means = model.means_
		print "Predicting"
		self.labels = model.predict(training_log)
		print self.means.shape
		print "Fit time: " + str(time.time() - start)

	def predict(self, testfile = 'test.csv', outfile = 'mixture_results.csv'):
		users = util.dict_from_csv("profiles_dict.csv")
		artists = util.dict_from_csv("artists_dict.csv")

		res = open(outfile, 'wb')
		writer = csv.writer(res)
		writer.writerow(['Id','plays'])

		testing = csv.reader(open(testfile, 'r'))
		next(testing, None)
		print "Getting values"
		for row in testing:
			if not row[0] % 1000:
				print row[0]
			user = int(users[row[1]])
			artist = int(artists[row[2]])
			label = int(self.labels[user])
			val = np.exp(self.means[label][artist])
			writer.writerow(row[i], val)

gM = gMixture()
gM.fit()
gM.predict()