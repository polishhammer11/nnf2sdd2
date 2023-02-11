from collections import defaultdict

filename = "house-votes-84.data"
new_filename = "house-votes-84-complete.data"

with open(filename,'r') as f:
    lines = f.readlines()

lines = [ line.strip().split(',') for line in lines ]
n = len(lines[0])


counts = {}
counts['republican'] = [ defaultdict(lambda: 0) for _ in range(n) ]
counts['democrat']   = [ defaultdict(lambda: 0) for _ in range(n) ]

for line in lines:
    party = line[0]
    for i,val in enumerate(line):
        counts[party][i][val] += 1

mappings = {}
mappings['republican'] = [ dict() for _ in range(n) ]
mappings['democrat']   = [ dict() for _ in range(n) ]

for party in counts:
    for count,mapping in zip(counts[party],mappings[party]):
        for key in count:
            if key == '?':
                if count['y'] >= count['n']:
                    mapping[key] = 'y'
                else:
                    mapping[key] = 'n'
                #mapping[key] = 'n'
            else:
                mapping[key] = key

with open(new_filename,'w') as f:
    for line in lines:
        party = line[0]
        line = [ mapping[key] for key,mapping in zip(line,mappings[party]) ]
        f.write(",".join(line))
        f.write("\n")
