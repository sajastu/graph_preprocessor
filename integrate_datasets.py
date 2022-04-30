import json
import pickle
import pandas as pd
from tqdm import tqdm

graph_files = {}

splits = {
    'train': ['barolo', 'brunello', 'chianti', 'barbaresco'],
    'dev': ['barbaresco', 'brunello'],
    'test': ['chianti', 'barolo'],
}

# for se in ['train', 'dev']:
for se in [ 'test']:
    instances = {}
    with open(f'/disk1/sajad/gov-reports/gov-reports/{se}-withIds.json') as fR:
        for l in fR:
            instances[json.loads(l)['doc_id']] = json.loads(l)

    graph_data = []
    for split in tqdm(splits[se], desc=f'{se}'):
        with open(f'/disk1/sajad/gov-reports/graph/{se}.graph.part.{split}.adj.pk', mode='rb') as fR:
            graph_data_split = pickle.load(fR)
            graph_data.extend(graph_data_split)

        if se == 'dev':
            with open(f'/disk1/sajad/gov-reports/graph/dev.single.graph.part.adj.pk', mode='rb') as fR:
                graph_data_split = pickle.load(fR)
                graph_data.extend(graph_data_split)

    graph_data_ori = []
    seen = []
    for g in graph_data:
        if g['doc_id'] not in seen:
            print(g['doc_id'])
            graph_data_ori.append(g)
            seen.append(g['doc_id'])


    assert len(graph_data_ori) == len(instances), "disc in len"

    new_instances = {
        'doc_id': [],
        'graph': []
    }

    for graph in graph_data:
        graph_files[graph['doc_id']] = graph



    with open(f'/disk1/sajad/gov-reports/graph/splits/{se}.graph.adj.pk', mode='wb') as fW:
        pickle.dump(graph_files, fW)

    graph_files = {}


################################################

# with open(f'/disk1/sajad/gov-reports/graph/train.ax.graph.adj.pk', mode='rb') as fR:
#     graph_data = pickle.load(fR)
#
# with open('/disk1/sajad/gov-reports/gov-reports/{se}-withIds-sample.json', mode='w') as fW:
#     for gkey, ggraph in graph_data.items():
#         doc_id = gkey
#         json.dump(instances[doc_id], fW)
#         fW.write('\n')