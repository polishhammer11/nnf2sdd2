from collections import defaultdict

with open("digits-.002/output.log") as f:
    lines = f.readlines()

counts = defaultdict(list)
for line in lines:
    line = line.strip()
    if len(line) < 2: continue
    if line[2] == '-':
        print( line )
        i,j = line.split('-')
        i,j = int(i),int(j)
        continue
    if line.startswith("#Exp"):
        #print(line)
        _,count = line.split(' ')
        count = int(count)
        counts[(i,j)].append(count)


