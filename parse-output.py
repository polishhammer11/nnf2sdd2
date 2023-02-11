from collections import defaultdict

with open("digits-.002/output.log") as f:
    lines = f.readlines()

counts = defaultdict(list)
for line in lines:
    line = line.strip()
    if len(line) < 2: continue
    if line[2] == '-':
        #print( line )
        i,j = line.split('-')
        i,j = int(i),int(j)
        continue
    if line.startswith("#Exp"):
        #print(line)
        _,count = line.split(' ')
        count = int(count)
        counts[(i,j)].append(count)


alist = defaultdict(list)
for key in counts:
    i,j = key
    print(i,"-",j)
    astar,depth,naive = counts[key]
    adivd = depth/astar
    adivn = naive/astar
    ddivn = naive/depth
    stats = adivd,adivn,ddivn
    print(adivd)
    alist[key].append(stats)

x=list(alist.values())
x.sort(reverse=True)
x=x[:3]
for i in x:
    for j in alist.keys():
        if(alist[j]==i):
            print(j,alist[j])
        
