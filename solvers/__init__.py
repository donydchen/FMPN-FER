"""
Created on Dec 13, 2018
@author: Yuedong Chen
"""

from .base_solver import BaseSolver
from .res_cls_solver import ResFaceClsSolver
from .res_solver import ResFaceSolver



def create_solver(opt):
    if opt.solver == 'res_cls':
        instance = ResFaceClsSolver()
    elif opt.solver == 'resface':
        instance = ResFaceSolver()
    else:
        instance = BaseSolver()

    instance.initialize(opt)
    return instance
