import json

for se in ['train', 'test', 'val']:
    n_instances = []
    with open(f'/disk1/sajad/gov-reports/gov-reports/{se}.json') as fR:
        for j, l in enumerate(fR):
            ent = json.loads(l)
            ent['doc_id'] = f'{se}-{j}'
            n_instances.append(ent)

    with open(f'/disk1/sajad/gov-reports/gov-reports/{se}-withIds.json', mode='w') as fW:
        for n in n_instances:
            json.dump(n, fW)
            fW.write('\n')