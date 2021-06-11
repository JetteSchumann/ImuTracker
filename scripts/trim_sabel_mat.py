import sys
import os
import scipy.io as sio
import numpy as np

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        #elif isinstance(elem,np.ndarray):
        #    dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem
    return dict

def trim_sabel_data_mat(file_name, start, end):
    if not os.path.exists(file_name):
        print(file_name, 'does not exist!')
        sys.exit()

    file, extension = os.path.splitext(file_name)
    new_file = file + '_trimmed_' + str(start) + '-' + str(end)
    mat_contents = sio.loadmat(file_name, struct_as_record=False, squeeze_me=True)

    mat_contents = _check_keys(mat_contents)

    mat_contents['athdata']['data'] = mat_contents['athdata']['data'][start - 1:end, :]
    sio.savemat(new_file, mat_contents)

    print('created ' + new_file)


if __name__ == "__main__":
    file_name = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    trim_sabel_data_mat(file_name, start, end)
