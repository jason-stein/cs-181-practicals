import csv
import time
import numpy as np

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
	np.save('train' + "_matrix", big_ass_matrix)
	return big_ass_matrix
# util.create_dict_csv('artists')
# util.create_dict_csv('profiles')