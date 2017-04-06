import csv
import time
import numpy as np
import requests
import json

def dict_from_csv(direc):
    dic = {}
    sys_map = csv.reader(open(direc, 'r'))
    next(sys_map, None)
    for row in sys_map:
        dic[str(row[0])] = row[1]
    return dic

def country_dict_from_csv(file, normalize=True):
	# Latitude runs between -90 and 90, longitude runs between -180 and 180
	dic = {}
	sys_map = csv.reader(open(file, 'r'))
	for row in sys_map:
		ll = eval(row[1])
		if normalize:
			ll = [ll[0]/90., ll[1]/180.]
		dic[str(row[0])] = ll
	return dic

def create_dict_csv(file):
	keys = dict_from_csv(file + ".csv").keys()
	filename = "{}_dict.csv".format(file)
	with open(filename, 'wb') as csv_file:
	    writer = csv.writer(csv_file)
	    writer.writerow(['none','sense'])
	    for i, key in enumerate(keys):
	    	writer.writerow([key, i])

def create_big_ass_matrix(filename):
	artists_dict = dict_from_csv('artists_dict.csv')
	profiles_dict = dict_from_csv('profiles_dict.csv')

	big_ass_matrix = np.zeros((len(profiles_dict), len(artists_dict)))

	start = time.time()
	with open(filename, 'r') as f:
		data = csv.reader(f, delimiter=',', quotechar='"')
		for line in data:
			if line[2].isdigit():
				row = int(profiles_dict[line[0]])
				col = int(artists_dict[line[1]])
				val = line[2]
				big_ass_matrix[row,col] = val

	print "Time to create big_ass_matrix: {}".format(time.time() - start)
	np.save('train' + "_matrix", big_ass_matrix)
	return big_ass_matrix

def create_test_train(filename, test_amt=.2):
	artists_dict = dict_from_csv('artists_dict.csv')
	profiles_dict = dict_from_csv('profiles_dict.csv')

	start = time.time()
	train = np.zeros((len(profiles_dict), len(artists_dict)))
	test = []

	with open(filename, 'r') as f:
		line_iter = csv.reader(f, delimiter=',', quotechar='"')
		data = list(line_iter)[1:]
		n = len(data)
		np.random.shuffle(data)
		test = data[:int(round(n*test_amt))]
		for t in test:
			t.insert(0, 1) # because normally we have indices in front of the test data
		data = data[int(round(n*test_amt)):]
		for line in data:
			row = int(profiles_dict[line[0]])
			col = int(artists_dict[line[1]])
			val = line[2]
			train[row, col] = val
	print np.array(test).shape
	print np.array(train).shape
	print "Time to create test and train: {}".format(time.time() - start)
	return test, train


def create_profile_matrix():
	start = time.time()
	countries = country_dict_from_csv('countries_latlng.csv')
	profiles_dict = dict_from_csv('profiles_dict.csv')
	
	profiles_matrix = np.zeros((len(profiles_dict), 3+2+3)) #m/f/unknown + age/unknown + lat/lng/unknown
	with open('profiles.csv', 'r') as f:
		data = csv.reader(f, delimiter=',', quotechar='"')
		next(data, None)
		for line in data:
			i = int(profiles_dict[line[0]])
			if line[1] == 'm':
				profiles_matrix[i][0] = 1
			elif line[1] == 'f':
				profiles_matrix[i][1] = 1
			else:
				profiles_matrix[i][2] = 1
			if line[2]:# and int(line[2]) > 10 and int(line[2]) < 100:
				profiles_matrix[i][3] = int(line[2])/80.
			else:
				profiles_matrix[i][4] = 1
			if line[3]:
				try:
					np.put(profiles_matrix[i], [5,6], countries[line[3]])
				except Exception as inst:
					print inst
					profiles_matrix[i][7] = 1
			else:
				profiles_matrix[i][7] = 1
	print "Time to create profiles_matrix: {}".format(time.time() - start)
	return profiles_matrix

def create_country_dict():
	countries = {}
	url_base = "https://maps.googleapis.com/maps/api/geocode/json?address={}"
	failed = []
	with open('profiles.csv', 'r') as f:
		data = csv.reader(f, delimiter=',', quotechar='"')
		next(data, None)
		i = 0
		for line in data:
			if not i % 10000:
				print i
			i += 1
			if line[3] in countries.keys() or line[3] in failed:
				continue
			try:
				response = requests.get(url_base.format(line[3]))
				resp = json.loads(response.content)
				lat = resp['results'][0]['geometry']['location']['lat']
				lng = resp['results'][0]['geometry']['location']['lng']
				countries[line[3]] = [lat, lng]
			except Exception as inst:
				print inst
				print line[3]
				failed.append(line[3])

	print countries
	with open('countries_latlng.csv', 'wb') as g:
		w = csv.writer(g)
		w.writerows(countries.items())

	print failed


# util.create_dict_csv('artists')
# util.create_dict_csv('profiles')
#create_profile_matrix()
