from .base_model import BaseModel
from .res_baseline import ResGenModel
from .res_cls import ResClsModel



def create_model(opt):
    # specify model name here
    if opt.model == "res_baseline":
        instance = ResGenModel()
    elif opt.model == "res_cls":
        instance = ResClsModel()
    else:
        instance = BaseModel()
    instance.initialize(opt)
    instance.setup()
    return instance
