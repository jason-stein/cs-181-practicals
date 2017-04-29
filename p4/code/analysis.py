from matplotlib import pyplot as plt

f = open("vel.txt", "r")
grav1 = []
grav4 = []
g1 = False
g1hold = None
g4hold = None

for line in f:
	if "Gravity 10" in line:
		if g1hold:
			grav1.append(g1hold)
		g1hold = []
		g1 = True
		continue
	elif "Gravity 40" in line:
		if g4hold:
			grav4.append(g4hold)
		g4hold = []
		g1 = False
		continue
	elif line == "\n":
		continue
	if g1:
		g1hold.append(int(line))
	else:
		g4hold.append(int(line))

f.close()

g1diffs = []
g4diffs = []

for l in grav1:
	prev = 0
	for i in l:
		g1diffs.append(abs(prev - i))
		prev = i

for l in grav4:
	prev = 0
	for i in l:
		g4diffs.append(abs(prev - i))
		prev = i

bins = range(0,50)
plt.hist(g1diffs, bins=bins)
plt.title("Velocity Changes For Gravity 1")
plt.show()

plt.hist(g4diffs, bins=bins)
plt.title("Velocity Changes For Gravity 4")
plt.show()

f = open("tredist.txt", "r")

treedists = []

for line in f:
	treedists.append(int(line))

plt.hist(treedists, bins=range(-150,500,10))
plt.title("Tree Distances")
plt.show()
