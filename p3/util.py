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
	return big_ass_matrix

def create_profile_matrix(filename):
	countries = set()
	with open(filename, 'r') as f:
		data = csv.reader(f, delimiter=',', quotechar='"')
		for line in data:
			countries.add(line[3])
	print len(countries)
	print countries

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
#create_profile_matrix('profiles.csv')
create_country_dict()