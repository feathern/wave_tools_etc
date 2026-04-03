import pickle
def read_tlon(idir,q,r,suffix):
    qstr = str(q)
    rstr = str(r)
    infile = idir+'/q'+qstr+'_r'+rstr+'_'+suffix
    with open(infile, 'rb') as file:
    # Load the data from the pickle file
        data_tuple = pickle.load(file)
    return data_tuple
def read_time_lon(infile):
    with open(infile, 'rb') as file:
    # Load the data from the pickle file
        data_tuple = pickle.load(file)
    return data_tuple