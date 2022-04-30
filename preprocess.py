import argparse
import os
from multiprocessing import cpu_count

from path_vars import DS_DIR, BASE_DIR
from preprocess_utils.conceptnet import extract_english, construct_graph

from preprocess_utils.convert_obqa import convert_to_obqa_statement

from preprocess_utils.convert_csqa import convert_to_entailment
from preprocess_utils.grounding import create_matcher_patterns, ground
from preprocess_utils.graph import generate_adj_data_from_grounded_concepts__use_LM

# from preprocess_utils.convert_csqa_vis import convert_to_entailment
# from preprocess_utils.graph_vis import generate_adj_data_from_ground_concepts__use_LM
# from preprocess_utils.grounding_vis import create_matcher_patterns, ground

# SERVER_NAME = os.environ['SERVER_NAME']
# if SERVER_NAME != 'barbaresco':
#     BASE_DIR = '/disk1/sajad/'
# else:
#     BASE_DIR = '/home/sajad/disk1/'
#
SERVER_NAME = 'aws'
DEB = False
DS_DIR='/cnn-dm/'
BASE_DIR='/dev/disk/data/'

if not DEB:

    input_paths = {
        'gov-reports': {
            'train': f'{BASE_DIR}/{DS_DIR}/train.json',
            'dev': f'{BASE_DIR}/{DS_DIR}/val.json',
            'test': f'{BASE_DIR}/{DS_DIR}/test.json',
        },
        'cpnet': {
            'csv': f'{BASE_DIR}/cpnet/conceptnet-assertions-5.6.0.csv',
        },
    }

    output_paths = {
        'cpnet': {
            'csv': f'{BASE_DIR}/cpnet/conceptnet.en.csv',
            'vocab': f'{BASE_DIR}/cpnet/concept.txt',
            'patterns': f'{BASE_DIR}/cpnet/matcher_patterns.json',
            'unpruned-graph': f'{BASE_DIR}/cpnet/conceptnet.en.unpruned.graph',
            'pruned-graph': f'{BASE_DIR}/cpnet/conceptnet.en.pruned.graph',
        },
        'gov-reports': {
            'statement': {
                'train': f'{BASE_DIR}/{DS_DIR}/statement/train.statement.part.{SERVER_NAME}.jsonl',
                'dev': f'{BASE_DIR}/{DS_DIR}/statement/dev.single.statement.part.{SERVER_NAME}.jsonl',
                'test': f'{BASE_DIR}/{DS_DIR}/statement/test.part.statement.part.{SERVER_NAME}.jsonl',
            },
            'ground': {
                'train': f'{BASE_DIR}/{DS_DIR}/ground/train.ground.part.{SERVER_NAME}.jsonl',
                'dev': f'{BASE_DIR}/{DS_DIR}/ground/dev.ground.part.{SERVER_NAME}.jsonl',
                'test': f'{BASE_DIR}/{DS_DIR}/ground/test.ground.part.{SERVER_NAME}.jsonl',
            },
            'graph': {
                'adj-train': f'{BASE_DIR}/{DS_DIR}/graph/train.graph.part.{SERVER_NAME}.adj.pk',
                'adj-dev': f'{BASE_DIR}/{DS_DIR}/graph/dev.single.graph.part.{SERVER_NAME}.adj.pk',
                'adj-test': f'{BASE_DIR}/{DS_DIR}/graph/test.graph.part.{SERVER_NAME}.adj.pk',
            },
        },
    }

else:
    input_paths = {
        'gov-reports': {
            'train': f'{BASE_DIR}/gov-reports/gov-reports/train-withIds.json',
            'dev': f'{BASE_DIR}/gov-reports/gov-reports/dev-withIds.json',
            'test': f'{BASE_DIR}/gov-reports/gov-reports/test-withIds.json',
        },
        'cpnet': {
            'csv': f'{BASE_DIR}/GreaseLM/data/cpnet/conceptnet-assertions-5.6.0.csv',
        },
    }

    output_paths = {
        'cpnet': {
            'csv': f'{BASE_DIR}/gov-reports/cpnet/conceptnet.en.csv',
            'vocab': f'{BASE_DIR}/gov-reports/cpnet/concept.txt',
            'patterns': f'{BASE_DIR}/gov-reports/cpnet/matcher_patterns.json',
            'unpruned-graph': f'{BASE_DIR}/gov-reports/cpnet/conceptnet.en.unpruned.graph',
            'pruned-graph': f'{BASE_DIR}/gov-reports/cpnet/conceptnet.en.pruned.graph',
        },
        'gov-reports': {
            'statement': {
                'train': f'{BASE_DIR}/gov-reports/statement/train.statement.debug.jsonl',
                'dev': f'{BASE_DIR}/gov-reports/statement/dev.statement.debug.jsonl',
                'test': f'{BASE_DIR}/gov-reports/statement/test.statement.debug.jsonl',
            },
            'ground': {
                'train': f'{BASE_DIR}/gov-reports/ground/train.ground.debug.jsonl',
                'dev': f'{BASE_DIR}/gov-reports/ground/dev.ground.debug.jsonl',
                'test': f'{BASE_DIR}/gov-reports/ground/test.ground.debug.jsonl',
            },
            'graph': {
                'adj-train': f'{BASE_DIR}/gov-reports/graph/train.graph.reduced.adj.pk',
                'adj-dev': f'{BASE_DIR}/gov-reports/graph/dev.graph.part.adj.pk',
                'adj-test': f'{BASE_DIR}/gov-reports/graph/test.graph.part.chianti.adj.pk',
            },
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common','cnndm'], choices=['cnndm'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            # {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'cnndm': [

            {'func': convert_to_entailment, 'args': (input_paths['gov-reports']['train'], output_paths['gov-reports']['statement']['train'])},
            # {'func': convert_to_entailment, 'args': (input_paths['gov-reports']['dev'], output_paths['gov-reports']['statement']['dev'])},
            # {'func': convert_to_entailment, 'args': (input_paths['gov-reports']['test'], output_paths['gov-reports']['statement']['test'])},

            {'func': ground, 'args': (output_paths['gov-reports']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['gov-reports']['ground']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['gov-reports']['statement']['dev'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['gov-reports']['ground']['dev'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['gov-reports']['statement']['test'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['gov-reports']['ground']['test'], args.nprocs)},

            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['gov-reports']['ground']['train'], output_paths['cpnet']['pruned-graph'],
                                                                                output_paths['cpnet']['vocab'], output_paths['gov-reports']['graph']['adj-train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['gov-reports']['ground']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['gov-reports']['graph']['adj-dev'], args.nprocs)},

            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['gov-reports']['ground']['test'],
            #                                                                     output_paths['cpnet']['pruned-graph'],
            #                                                                     output_paths['cpnet']['vocab'],
            #                                                                     output_paths['gov-reports']['graph']['adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
