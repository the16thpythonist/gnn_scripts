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

from code import *

PATH = pathlib.Path(__file__).parent.absolute()
CODE_PATH = os.path.join(PATH, 'code.py')
experiment = Experiment.load(CODE_PATH)
experiment.analyses = []


# == __main__.analysis ==
@experiment.analysis
def analysis(e: Experiment):
    e.log('starting analysis...')


experiment.execute_analyses()