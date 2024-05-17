import json
import commentjson
import pickle
import scipy.io as sio
from pathlib import Path
import numpy as np


def read_commentjson(path):
    with open(path, mode="r") as f:
        return commentjson.load(f)


def load_json(path):
    """The function of loading json file

    Arguments:
        path {str} -- The path of the json file

    Returns:
        list, dict -- The obj stored in the json file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def dump_json(path, data):
    """Dump data to json file

    Arguments:
        data {[Any]} -- data
        path {str} -- json file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def load_text(path):
    with open(path) as f:
        return [l.replace("\n", "") for l in f.readlines()]


def write_pkl(data, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


### ---------------------------------  load BFM data
def load_BFM(model_path):
    ''' load BFM 3DMM model
    Args:
        model_path: path to BFM model. 
    Returns:
        model: (nver = 53215, ntri = 105840). nver: number of vertices. ntri: number of triangles.
            'shapeMU': [3*nver, 1]
            'shapePC': [3*nver, 199]
            'shapeEV': [199, 1]
            'expMU': [3*nver, 1]
            'expPC': [3*nver, 29]
            'expEV': [29, 1]
            'texMU': [3*nver, 1]
            'texPC': [3*nver, 199]
            'texEV': [199, 1]
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++)
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles)
            'kpt_ind': [68,] (start from 1)
    PS:
        You can change codes according to your own saved data.
        Just make sure the model has corresponding attributes.
    '''
    C = sio.loadmat(model_path)
    model = C['model']
    model = model[0, 0]

    # change dtype from double(np.float64) to np.float32,
    # since big matrix process(espetially matrix dot) is too slow in python.
    model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
    model['shapePC'] = model['shapePC'].astype(np.float32)
    model['shapeEV'] = model['shapeEV'].astype(np.float32)
    model['expEV'] = model['expEV'].astype(np.float32)
    model['expPC'] = model['expPC'].astype(np.float32)

    # matlab start with 1. change to 0 in python.
    model['tri'] = model['tri'].T.copy(order='C').astype(np.int32) - 1
    model['tri_mouth'] = model['tri_mouth'].T.copy(order='C').astype(
        np.int32) - 1
    # kpt ind
    model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

    return model


def load_uv_coords(path='BFM_UV.mat'):
    ''' load uv coords of BFM
    Args:
        path: path to data.
    Returns:  
        uv_coords: [nver, 2]. range: 0-1
    '''
    C = sio.loadmat(path)
    uv_coords = C['UV'].copy(order='C')
    return uv_coords
