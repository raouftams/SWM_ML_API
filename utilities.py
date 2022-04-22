import pickle
import numpy
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        return JSONEncoder.default(self, obj)


def openPkl(file_path):
    with  open(file_path,"rb") as file:
        return pickle.load(file)

def savePkl(objname,path):
    with  open(path,"wb") as file:
        pickle.dump(objname,file,pickle.HIGHEST_PROTOCOL)