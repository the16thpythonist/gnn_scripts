"""
This is the base experiments which implements the common functionality that is required 
for all the different model trainings. That mainly includes:

- The loading and pre-processing of the dataset.
- The visualization of the classification results.

This file should *not* be executed by itself.
"""
import os
import csv
import random
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

PATH = pathlib.Path(__file__).parent.absolute()

# == DATASET PARAMETERS ==
# All the parameters which are related to the dataset, which includes the loading of the raw 
# dataset data from the CSV file, but also the train test split for the model training.

# param CSV_PATH:
#       The absolute path path pointing to the CSV file containing the SMILES codes and the 
#       corresponding ground truth class labels.
CSV_PATH: str = os.path.join(PATH, 'assets', 'test.csv')
# param SMILES_COLUMN_NAME:
#       In the dataset csv file, the string title of the column which contains the SMILES
#       representations of the elements.
SMILES_COLUMN_NAME: str = 'smiles'
# param TARGET_COLUMN_NAME:
#       In the dataset csv file, the string title of the column which contains the annotated 
#       class label. This label has to be represented as either a "1" or a "0"
TARGET_COLUMN_NAME: str = 'target'
# param NUM_TEST:
#       The number of the elements of the dataset to be randomly chosen as the test set 
#       all other elements will automatically be assigned as the train set.
NUM_TEST: int = 1000

# == EVALUATION PARAMETERS ==
LOG_STEP: int = 1000
PLOT_COLOR: str = 'blue'
PLOT_CMAP: str = 'Blues'

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
                'target': int(row[TARGET_COLUMN_NAME]), 
            }
            
            # we'll only test with a very small number of elements
            if __TESTING__ and index > 2000:
                break
            
    dataset_length = len(index_data_map)
    e.log(f'loaded {dataset_length} elements from CSV')
        
    # ~ DATA PROCESSING
    
    e.log('processing the smiles into graph representations...')
    processing = MoleculeProcessing()
    for c, (index, data) in enumerate(index_data_map.items()):
        graph = processing.process(data['smiles'])
        graph['graph_labels'] = [0, 1] if data['target'] else [1, 0]
        data['graph'] = graph
        
        if c % LOG_STEP == 0:
            e.log(f' * ({c}/{dataset_length}) done')
            
    indices = list(index_data_map.keys())
    test_indices = random.sample(indices, k=NUM_TEST)
    train_indices = list(set(indices).difference(set(test_indices)))
    e['test_indices'] = test_indices
    e['train_indices'] = train_indices

    # ~ MODEL TRAINING
    model = e.apply_hook(
        'train_model',
        index_data_map=index_data_map,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    
    # ~ MODEL EVALUATION
    e.apply_hook(
        'evaluate_model',
        model=model,
        index_data_map=index_data_map,
        train_indices=train_indices,
        test_indices=test_indices,
        processing=processing,
    )
    
    

@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')
    
    # ~ TRAINING LOSS VISUALIZATION
    if 'loss' in e.data:
        e.log('plotting the training loss...')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        epochs, values = zip(*enumerate(e['loss']))
        ax.plot(epochs, values, color=e.PLOT_COLOR)
        ax.set_title('Training Loss')
        ax.set_ylabel('CCE Loss')
        ax.set_xlabel('Epochs')
        ax.scatter(
            epochs[-1], values[-1],
            c=e.PLOT_COLOR,
            label=f'{values[-1]:.1f}'
        )
        ax.legend()
        fig_path = os.path.join(e.path, 'training_loss.pdf')
        fig.savefig(fig_path)
        plt.close()
        
    # ~ CONFUSION MATRIX
    if 'out' in e.data:
        test_indices = list(e['test_indices'])
        y_true = np.array(list(e['out/true'].values()))
        y_pred = np.array(list(e['out/pred'].values()))
        
        labels_true = [np.argmax(value) for value in y_true]
        labels_pred = [np.argmax(value) for value in y_pred]
        
        e['accuracy'] = accuracy_score(labels_true, labels_pred)
        e['f1'] = f1_score(labels_true, labels_pred)
        
        e.log('plotting confusion matrix...')
        cf = confusion_matrix(labels_true, labels_pred)
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            data=cf,
            cmap=e.PLOT_CMAP,
            annot=True,
        )
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        fig_path = os.path.join(e.path, 'confusion_matrix.pdf')
        plt.savefig(fig_path)
        plt.close()
    
    # ~ AUROC CURVE
    if 'out' in e.data:
        e.log('plotting ROC curve...')
        fpr, tpr, _ = roc_curve(y_true[:, 1], y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        e['auroc'] = roc_auc
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        ax.plot(fpr, tpr, color=e.PLOT_COLOR, label=f'ROC AUC: {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color=e.PLOT_COLOR, alpha=0.5, ls='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend()
        fig_path = os.path.join(e.path, 'roc.pdf')
        fig.savefig(fig_path)
    
    # ~ PRINT RESULTS
    metrics = ['auroc', 'f1', 'accuracy']
    e.log('printing results:')
    for metric in metrics:
        if metric in e.data:
            value = e[metric]
            e.log(f' * {metric}: {value:.3f}')
        
    e.save_data()
    

experiment.run_if_main()