import numpy as np
import pandas as pd
from scipy import ndimage

def tlags3d(
    df,
    use_stride_tricks=True,
    lags=[1,]
):

    '''
    tlag3d created 22/03/2021 by Jim Dale
    
    Performs temporal lags on a dataframe by transforming from flat format to 3d tensor 
    with dimensions time x pgid x features.
    
    Accepts arbitrary numbers of columns in input df, and arbitrary list of lags.
    
    Arguments:
    
    df:                a dataframe of series to be timelagged
    
    use_stride_tricks: boolean, decide to use stride_tricks or not (optional, 
                       defaults to True)
                       
    lags:              a list of lags to be applied (optional, defaults to [1,])
    
    Returns:
    
    A dataframe with all lags of all columns in input df
    
    '''


    df=df.fillna(0.0)
    df=df.sort_index()
    df_index=df.index

    times,time_to_index,index_to_time=map_times(df)

    features=map_features(df)

    pgids,pgid_to_index,index_to_pgid=map_pgids(df)

    if (use_stride_tricks):
        tensor3d=df_to_tensor_strides(df)
    else:
        tensor3d=df_to_tensor_no_strides(df)

    tlags=get_tlags(times,pgids,features,time_to_index,tensor3d,lags)

    df_tlags=tlags_to_df_strides(tlags,times,time_to_index,pgids,features,lags,df_index)

    return df_tlags


def df_to_tensor_strides(
    df
):
    '''
    df_to_tensor created 13/03/2021 by Jim Dale
    Uses as_strided from numpy stride_tricks library to create a tensorlike
    from a dataframe.
    
    There are two possibilities:
    
    (i) If the dataframe is uniformly-typed, the tensor is a memory view: 
    changes to values in the tensor are reflected in the source dataframe.

    (ii) Otherwise, the tensor is a copy of the dataframe and changes in it
    are not propagated to the dataframe.

    Note that dim0 of the tensor corresponds to level 0 of the df multiindex,
    dim1 corresponds to level 1 of the df multiindex, and dim2 corresponds to 
    the df's columns.
    '''

    # get shape of dataframe

    dim0,dim1=df.index.levshape
    
    dim2=df.shape[1]
    
    # check that df can in principle be tensorised
    
    if dim0*dim1!=df.shape[0]:
        raise Exception("df cannot be cast to a tensor - dim0 * dim1 != df.shape[0]",dim0,dim1,df.shape[0])

    flat=df.to_numpy()

    # get strides (in bytes) of flat array
    flat_strides=flat.strides

    offset2=flat_strides[1]

    offset1=flat_strides[0]
    
    # compute stride in bytes along dimension 0
    offset0=dim1*offset1

    # get memory view or copy as a numpy array
    tensor3d=np.lib.stride_tricks.as_strided(flat,shape=(dim0,dim1,dim2),
                                           strides=(offset0,offset1,offset2))
                                           
    return tensor3d

def df_to_tensor_no_strides(
    df
):
    
    # get shape of dataframe
    
    dim0,dim1=df.index.levshape
    
    dim2=df.shape[1]
    
    # check that df can in principle be tensorised
    
    if dim0*dim1!=df.shape[0]:
        raise Exception("df cannot be cast to a tensor - dim0 * dim1 != df.shape[0]",dim0,dim1,df.shape[0])

    
    flat=df.to_numpy()

    tensor3d=np.empty((dim0,dim1,dim2))
    chunksize=dim1
    nchunks=dim0
    for ichunk in range(nchunks):
        tensor3d[ichunk,:,:]=flat[ichunk*chunksize:(ichunk+1)*chunksize,:]

#    print(tensor3d[:,5550,:])

    return tensor3d
    
def map_times(
    df
):

    times=np.array(list({idx[0] for idx in df.index.values}))
    times=list(np.sort(times))
    
    time_to_index={}
    index_to_time={}
    for i,time in enumerate(times):
        time_to_index[time]=i
        index_to_time[i]=time
    
    return times,time_to_index,index_to_time
    
def map_pgids(
    df
):
    
    pgids=np.array(list({idx[1] for idx in df.index.values}))

    pgids=np.sort(pgids)
    
    pgid_to_index={}
    index_to_pgid={}

    for i,pgid in enumerate(pgids):
        pgid_to_index[pgid]=i
        index_to_pgid[i]=pgid

    return pgids,pgid_to_index,index_to_pgid
    
def map_features(
    df
):

    features=df.columns
    
    return features
    
    
def get_tlags(
    times,
    pgids,
    features,
    time_to_index,
    tensor3d,
    lags
):

    tlags=np.zeros((len(times),len(pgids),len(features)*len(lags)))

    for ifeature in range(len(features)):
        for ilag,lag in enumerate(lags):
            for time in times:
                tindex=time_to_index[time]            
                try:
                    tlags[tindex,:,ifeature*len(lags)+ilag]=tensor3d[tindex-lag,:,ifeature]
                except:
                    pass  

    return tlags
    
def tlags_to_df_no_strides(
    tensor3d,
    tlags,
    times,
    time_to_index,
    pgids,
    features,
    lags,
    df_index
):

    dim0,dim1,dim2=len(times),len(pgids),len(features)*len(lags)
    
    flat=np.empty((dim0*dim1,dim2))
    
    for ichunk in range(dim0):
        flat[ichunk*dim1:(ichunk+1)*dim1,:]=tensor3d[ichunk,:,:]
       
    df_column_names=['tlag_'+str(lag)+'_'+feature for feature in features for lag in lags]
        
    df_tlags=pd.DataFrame(flat, index=df_index, columns=df_column_names)
    
    return df_tlags    
    
    
def tlags_to_df_strides(
    tlags,
    times,
    time_to_index,
    pgids,
    features,
    lags,
    df_index
):

    dim0,dim1,dim2=len(times),len(pgids),len(features)*len(lags)
    
    flat=np.empty((dim0*dim1,dim2))
    
    tlags_strides=tlags.strides

    offset2=tlags_strides[2]

    offset1=tlags_strides[1]
    
    offset0=tlags_strides[0]
    
    flat=np.lib.stride_tricks.as_strided(tlags,shape=(dim0*dim1,dim2),
                                           strides=(offset1,offset2))
       
    df_column_names=['tlag_'+str(lag)+'_'+feature for feature in features for lag in lags]
        
    df_tlags=pd.DataFrame(flat, index=df_index, columns=df_column_names)
    
    return df_tlags    
    