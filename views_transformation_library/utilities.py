import numpy as np

def _df_to_tensor_no_strides(df):
    '''
    df_to_tensor_no_strides created 13/03/2021 by Jim Dale
    Uses regular array indexing to create a tensorlike from a dataframe.

    dim0 of the tensor corresponds to level 0 of the df multiindex,
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

    tensor3d=np.empty((dim0,dim1,dim2))
    chunksize=dim1
    nchunks=dim0
    for ichunk in range(nchunks):
        tensor3d[ichunk,:,:]=flat[ichunk*chunksize:(ichunk+1)*chunksize,:]

    return tensor3d 

def _df_to_tensor_strides(df):
    '''
    df_to_tensor created 13/03/2021 by Jim Dale
    Uses as_strided from numpy stride_tricks library to create a tensorlike
    from a dataframe.

    dim0 of the tensor corresponds to level 0 of the df multiindex,
    dim1 corresponds to level 1 of the df multiindex, and dim2 corresponds to
    the df's columns.
    '''

    # get shape of dataframe

    dim0,dim1=df.index.levshape

    dim2=df.shape[1]

    # check that df can in principle be tensorised

    if dim0*dim1!=df.shape[0]:
        raise Exception("df cannot be cast to a tensor - dim0 * dim1 != df.shape[0]",
                        dim0,dim1,df.shape[0])

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
    

def _map_times(df):
    '''
    
    map_times
    
    Gets the times from a dataframe, assuming that they are the first index, and returns
    a list of unique times, and dicts which convert between the actual time, and the
    index in the list.

    '''
    # get unique times

    times=np.array(list({idx[0] for idx in df.index.values}))
    times=list(np.sort(times))
    
    # make dicts to transform between times and the index of a time in the list
    
    time_to_index={}
    index_to_time={}
    for i,time in enumerate(times):
        time_to_index[time]=i
        index_to_time[i]=time
    
    return times,time_to_index,index_to_time
    
def _map_features(df):
    '''
    
    map_features
    
    Gets a list of features from a dataframe.

    '''

    features=list(df.columns)
    
    return features
    
def _map_pgids_1d(df):
    '''
    
    map_pgid_1d
    
    Gets the pgids from a dataframe, assuming that they are the second index, and 
    returns a 1d list of unique pgids, together with dicts which convert between 
    the actual pgid and the index in the list.
    
    '''
    
    pgids=np.array(list({idx[1] for idx in df.index.values}))

    pgids=np.sort(pgids)
    
    pgid_to_index={}
    index_to_pgid={}

    for i,pgid in enumerate(pgids):
        pgid_to_index[pgid]=i
        index_to_pgid[i]=pgid

    return pgids,pgid_to_index,index_to_pgid
    
    
def _map_pgids_2d(df):
    '''
	map_pgids_2d
	
	This function builds a 2D map in longitude-latitude from the pgids contained in
	the input dataframe assuming they are the second index column, and creates dicts 
	allowing quick transformation from (long,lat) to pgid and vice versa.
	
	The pgids are embedded and centred in the smallest possible square grid whose side
	is an integer power of 2

	'''
	
    PG_STRIDE=720
	
	# get unique pgids
	
    pgids=np.array(list({idx[1] for idx in df.index.values}))
    pgids=np.sort(pgids)
		
	# convert pgids to longitudes and latitudes
	
    longitudes=pgids%PG_STRIDE
    latitudes=pgids//PG_STRIDE
	
    latmin=np.min(latitudes)
    latmax=np.max(latitudes)
    longmin=np.min(longitudes)
    longmax=np.max(longitudes)

    latrange=latmax-latmin
    longrange=longmax-longmin

	# shift to a set of indices that starts at [0,0]

    latitudes-=latmin
    longitudes-=longmin

    # find smallest possible square grid with side 2^ncells which will fit the pgids

    latmin=np.min(latitudes)
    latmax=np.max(latitudes)
    longmin=np.min(longitudes)
    longmax=np.max(longitudes)

    maxsize=np.max((longrange,latrange))
    power=1+int(np.log2(maxsize))

    ncells=2**power

    # centre the pgids

    inudgelong=int((ncells-longmax)/2)
    inudgelat=int((ncells-latmax)/2)

    longitudes+=inudgelong
    latitudes+=inudgelat

	# make dicts to transform between pgids and (long,lat) coordinate

    pgid_to_longlat={}
    longlat_to_pgid={}
    
    pgid_to_index={}
    index_to_pgid={}

    for i,pgid in enumerate(pgids):
        pgid_to_longlat[pgid]=(longitudes[i],latitudes[i])
        longlat_to_pgid[(longitudes[i],latitudes[i])]=pgid
        pgid_to_index[pgid]=i
            
    return pgids,pgid_to_longlat,longlat_to_pgid,pgid_to_index,index_to_pgid,ncells,power
    
def _build_4d_tensor(
    df,
    pgids,
    pgid_to_index,
    times,
    time_to_index,
    longrange,
    latrange,
    pgid_to_longlat,
    features,
    use_stride_tricks
):
    '''
    
    build_4d_tensor
    
    This function accepts an input dataframe and the dicts generated by the map_times,
    map_features and map_pgids_2d functions, and returns a 4d latitude x longitude x
    time x feature tensor.

    
    '''
    
    # convert flat data from df into time x pgid x feature tensor
    
    if(use_stride_tricks):
        tensor3d=_df_to_tensor_strides(df)
    else:
        tensor3d=_df_to_tensor_strides(df)

    # convert 3d tensor into longitude x latitude x time x feature tensor

    tensor4d=np.zeros((longrange,latrange,len(times),len(features)))
    for pgid in pgids:

        pgindex=pgid_to_index[pgid]
        for time in times:
        
            tindex=time_to_index[time]
            ilong=pgid_to_longlat[pgid][0]
            ilat=pgid_to_longlat[pgid][1]
            
            tensor4d[ilong,ilat,tindex,:]=tensor3d[tindex,pgindex,:]
                 
    return tensor4d    
    
def _df_to_datacube(df,use_stride_tricks):

    '''

    df_to_datacube
    
    This function streamlines the pgid, time and feature mapping, together with the 4d
    tensor build, so that a df is accepted and a 4d tensor is produced to enable quick
    plotting.

    '''

    pgids,pgid_to_longlat,longlat_to_pgid,pgid_to_index,index_to_pgid,ncells,power=_map_pgids_2d(df)

    times,time_to_index,index_to_time=_map_times(df)

    features=_map_features(df)
    
    tensor=_build_4d_tensor(
                           df,
                           pgids,
                           pgid_to_index,
                           times,
                           time_to_index,
                           ncells,
                           ncells,
                           pgid_to_longlat,
                           features,
                           use_stride_tricks)
    
    return tensor    
        