"""
This string will be saved to the experiment's archive folder as the "experiment description"

CHANGELOG

0.1.0 - initial version
"""
import os
import csv
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.processing.molecules import MoleculeProcessing

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
CSV_PATH: str = os.path.join(PATH, 'assets', 'test.csv')
SMILES_COLUMN_NAME: str = 'smiles'
TARGET_COLUMN_NAME: str = 'aggregator'

# == EVALUATION PARAMETERS ==
LOG_STEP: int = 1000

__DEBUG__ = True
__TESTING__ = False

@Experiment(base_path=folder_path(__file__),
            namespace=file_namespace(__file__),
            glob=globals())
def experiment(e: Experiment):
    e.log('starting experiment...')

    # ~ DATA LOADING
    e.log('loading dataset from CSV...')
    index_data_map = {}
    with open(CSV_PATH, mode='r') as file:
        dict_reader = csv.DictReader(file)
        for index, row in enumerate(dict_reader):
            index_data_map[index] = {
                'smiles': row[SMILES_COLUMN_NAME],
                'target': row[TARGET_COLUMN_NAME], 
            }
            
    dataset_length = len(index_data_map)
    e.log(f'loaded {dataset_length} elements from CSV')
        
    # ~ DATA PROCESSING
    
    e.log('processing the smiles into graph representations...')
    processing = MoleculeProcessing()
    for c, (index, data) in enumerate(index_data_map.items()):
        graph = processing.process(data['smiles'])
        graph['graph_labels'] = [0, 1] if data['target'] else [1, 0]
        
        if c % LOG_STEP == 0:
            e.log(f' * ({c}/{dataset_length}) done')

    

@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.run_if_main()