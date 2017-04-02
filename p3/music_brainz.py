import requests
import csv

artists_file = 'artists.csv'
with open(artists_file, 'r') as artists:
	artists = csv.reader(artists, delimiter=',', quotechar='"')
	next(artists, None)
	for artist in artists:
		print artist
		ID = artist[0]
		url = 'https://musicbrainz.org/ws/2/artist/' + ID + '?inc=tags&fmt=json'
		r = requests.get(url)
		# if 503 keep trying
		while r.status_code == 503:
			r = requests.get(url)
		# once we successfully get the response
		if r.status_code == 200:
			tags = r.json()['tags']
			for tag in tags:
				if tag['count'] != 0:
					print tag['name']