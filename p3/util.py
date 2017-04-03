import csv

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
	    
# util.create_dict_csv('artists')
# util.create_dict_csv('profiles')