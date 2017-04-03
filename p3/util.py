import csv

def dict_from_csv(direc):
    dic = {}
    sys_map = csv.reader(open(direc, 'r'))
    for row in sys_map:
        dic[str(row[0])] = row[1]
    return dic

def create_dict_csv(file):
	keys = dict_from_csv(file + ".csv").keys()
	filename = "{}_dict.csv".format(file)
	with open(filename, 'wb') as csv_file:
	    writer = csv.writer(csv_file)
	    for i, key in enumerate(keys):
	    	writer.writerow([i, key])
	    