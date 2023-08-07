
"""
This is a sub experiment derived from ...

CHANGELOG

0.1.0 - initial version
"""
import os
import pathlib
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

PATH = pathlib.Path(__file__).parent.absolute()


# == MODEL PARAMETERS ==
UNITS: t.List[int] = [64, 64, 64]
FINAL_UNITS: t.List[int] = [64, 32, 2]

EPOCHS: int = 50
BATCH_SIZE: int = 16

# == MODEL IMPLEMENTATION ==

class GatModel(ks.models.Model):
    
    def __init__(self, 
                 units: t.List[int],
                 final_units: t.List[int],
                 activation: str = 'kgcnn>leaky_relu',
                 pooling_method: str = 'sum',
                 final_activation: str = 'softmax',
                 **kwargs):
        super(GatModel, self).__init__(**kwargs)
        
        self.units = units
        self.final_units = final_units
        self.activation = activation
        self.pooling_method = pooling_method
        
        self.conv_layers = []
        for k in self.units:
            lay = AttentionHeadGATV2(
                units=k,
                activation=self.activation,
                use_edge_features=True,
            )
            self.conv_layers.append(lay)
            
        self.lay_pooling = PoolingNodes(pooling_method='sum')
        
        self.final_layers = []
        self.final_acts = ['relu' for _ in self.final_units]
        self.final_acts[-1] = final_activation
        for k, act in zip(self.final_units, self.final_acts):
            lay = DenseEmbedding(
                units=k,
                activation=act,
            )
            self.final_layers.append(lay)
            
        
    def call(self, inputs, **kwargs):
        node_input, edge_input, edge_indices = inputs
        
        node_embeddings = node_input
        for lay in self.conv_layers:
            node_embeddings = lay([node_embeddings, edge_input, edge_indices])

        graph_embeddings = self.lay_pooling(node_embeddings)
        out = graph_embeddings
        for lay in self.final_layers:
            out = lay(out)
        
        return out
    
    
__DEBUG__ = True
__TESTING__ = True


experiment = Experiment.extend(
    'train_model.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('train_model')
def train_model(e: Experiment,
                index_data_map: dict,
                train_indices: t.List[int],
                test_indices: t.List[int]):
    
    # For the testing we only do a few epochs
    if e.__TESTING__:
        e.EPOCHS = 3
    
    # ~ DATA PREPARATION
    # Before training the model we first have to convert the dataset which is currently a list of graph dictionaries 
    # into the tensors that tensorflow expects for the model
    graphs_train = [index_data_map[i]['graph'] for i in train_indices]
    x_train = (
        ragged_tensor_from_nested_numpy([graph['node_attributes'] for graph in graphs_train]),
        ragged_tensor_from_nested_numpy([graph['edge_attributes'] for graph in graphs_train]),
        ragged_tensor_from_nested_numpy([graph['edge_indices'] for graph in graphs_train])
    )
    y_train = np.array([graph['graph_labels'] for graph in graphs_train])
    
    # ~ MODEL TRAINING
    e.log('starting model_training')
    model = GatModel(
        units=e.UNITS,
        final_units=e.FINAL_UNITS,
    )
    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=1e-3),
        loss=ks.losses.CategoricalCrossentropy(),
    )
    hist = model.fit(
        x_train, y_train,
        epochs=e.EPOCHS,
        batch_size=e.BATCH_SIZE,
    )
    
    # Now we save the progression of the loss over the epochs into the experiment storage so that it can be visualized 
    # during the analysis of the experiment
    e['loss'] = hist.history['loss']
    
    # at the end we have to return the trained model
    return model


@experiment.hook('evaluate_model')
def evaluate_model(e: Experiment,
                   model: ks.models.Model,
                   index_data_map: dict,
                   train_indices: list,
                   test_indices: list,
                   **kwargs):
    e.log('model evaluation...')
    
    graphs_test = [index_data_map[i]['graph'] for i in test_indices]
    x_test = (
        ragged_tensor_from_nested_numpy([graph['node_attributes'] for graph in graphs_test]),
        ragged_tensor_from_nested_numpy([graph['edge_attributes'] for graph in graphs_test]),
        ragged_tensor_from_nested_numpy([graph['edge_indices'] for graph in graphs_test]),
    )
    y_test = [graph['graph_labels'] for graph in graphs_test]
    
    e.log('making test predictions...')
    y_pred = model(x_test)
    # we'll also want to save the raw predictions into the experiment storage so that we can compute the 
    # evaluation metrics later on during the analysis step
    for c, index in enumerate(test_indices):
        e[f'out/true/{index}'] = y_test[c]
        e[f'out/pred/{index}'] = y_pred[c]

experiment.run_if_main()