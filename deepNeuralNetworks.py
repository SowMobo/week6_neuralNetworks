# Importing dependencies
from matplotlib.pyplot import cla
import numpy as np
from code_for_hw7 import *
import modules_disp as disp

# Super class `Module`
class Module:
    def sgd_step(self, lrate): # for modules w/o weights
        pass
    