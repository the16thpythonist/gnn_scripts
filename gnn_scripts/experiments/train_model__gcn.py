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


# == MODEL PARAMETERS ==
# All the parameters that are relevant to the model training process, which includes 
# the architecture configuration of the model and the trainin hyperparameters

# :param UNITS:
#       A list that determines the number of hidden units in each of the message-passing layers 
#       of the network. Each elements represents layers - adding and removing elements from this list 
#       will result in the addition and removal of the corresponding layer.
UNITS: t.List[int] = [64, 64, 64]
# :param FINAL_UNITS:
#       A list that determines the number of hidden units in each dense layer of the final 
#       prediction part of the network. The last value of this list has to be 2 since that represents 
#       the model output for the two classes involved in the binary classification problem.
FINAL_UNITS: t.List[int] = [64, 32, 2]
# :param EPOCHS:
#       The number of epochs for which to train the model
EPOCHS: int = 50
# :param BATCH_SIZE:
#       The number of elements from the train set to show to the model at once during the training 
#       process.
BATCH_SIZE: int = 16

# == MODEL IMPLEMENTATION ==
# This class implements the actual model.

class GcnModel(ks.models.Model):
    
    def __init__(self, 
                 units: t.List[int],
                 final_units: t.List[int],
                 activation: str = 'kgcnn>leaky_relu',
                 pooling_method: str = 'sum',
                 final_activation: str = 'softmax',
                 **kwargs):
        super(GcnModel, self).__init__(**kwargs)
        
        self.units = units
        self.final_units = final_units
        self.activation = activation
        self.pooling_method = pooling_method
        
        self.conv_layers = []
        for k in self.units:
            lay = GCN(
                units=k,
                activation=self.activation,
                # use_edge_features=True,
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
        ragged_tensor_from_nested_numpy([[[1.0] for _ in graph['edge_indices']] for graph in graphs_train]),
        ragged_tensor_from_nested_numpy([graph['edge_indices'] for graph in graphs_train])
    )
    y_train = np.array([graph['graph_labels'] for graph in graphs_train])
    
    # ~ MODEL TRAINING
    e.log('starting model_training')
    model = GcnModel(
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
        ragged_tensor_from_nested_numpy([[[1.0] for _ in graph['edge_indices']] for graph in graphs_test]),
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