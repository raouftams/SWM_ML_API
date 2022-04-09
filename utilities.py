import pickle

def openPkl(file_path):
    with  open(file_path,"rb") as file:
        return pickle.load(file)

def savePkl(objname,path):
    with  open(path,"wb") as file:
        pickle.dump(objname,file,pickle.HIGHEST_PROTOCOL)