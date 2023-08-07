#! /usr/bin/env python3
"""
This python module was automatically generated.

This module can be used to perform analyses on the results of an experiment which are saved in this archive
folder, without actually executing the experiment again. All the code that was decorated with the
"analysis" decorator was copied into this file and can subsequently be changed as well.
"""
import os
import json
import pathlib
from pprint import pprint
from typing import Dict, Any

# Useful imports for conducting analysis
import numpy as np
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment

# Importing the experiment

from train_model import *

from experiment_code import *

PATH = pathlib.Path(__file__).parent.absolute()
CODE_PATH = os.path.join(PATH, 'experiment_code.py')
experiment = Experiment.load(CODE_PATH)
experiment.analyses = []


# == /media/ssd/Programming/gnn_scripts/gnn_scripts/experiments/train_model.analysis ==
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


experiment.execute_analyses()