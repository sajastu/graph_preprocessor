import json

for se in ['train', 'test', 'val']:
    counter=0
    instances = []
    with open(f'/disk1/sajad/datasets/medical/mental-reddit-reduced/sets/{se}.json') as fR:
        for l in fR:
            ent = json.loads(l.strip())
            if ent['subreddit'] == 'BPD':
                continue
            else:
                instances.append(ent)

                counter += 1
        print(counter)


    with open(f'/disk1/sajad/datasets/medical/mental-reddit-reduced/sets/{se}-final.json', mode='w') as fW:
        for ins in instances:
            json.dump(ins, fW)
            fW.write('\n')
