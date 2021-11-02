import numpy as np
import ingester3
from ingester3.Country import Country
from ingester3.scratch import fetch_data

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
    
def _get_country_neighbours_tensor():

    '''
    
    get_country_neighbours_tensor
    
    Fetches the country_country_month_expanded table and transforms it to a tensor with
    dimensions month x country x country, where (month_i, country_a, country_b)=1 
    indicates that country_a and country_b are neighbours in month_i. 
    
    Note that (month_i, country_a, country_a)=0, and that countries which never have any 
    neighbours do not appear in the tensor.
    
    '''

    columns = ['month_id', 'a_id', 'b_id']

    df_neighbs = fetch_data(loa_table='country_country_month_expanded', columns=columns)

    df_neighbs=df_neighbs[columns].sort_values(
                                               ["month_id", "a_id","b_id"], 
                                               ascending = (True, True, True)
                                               )

    neighb_month_ids=np.sort(df_neighbs['month_id'].unique())

    neighb_country_ids=np.sort(df_neighbs['b_id'].unique())

    nmonths=len(neighb_month_ids)
    
    ncountries=len(neighb_country_ids)

    neighb_data=np.zeros((nmonths,ncountries,ncountries))
    
    neighb_tensor_dims=['month', 'country', 'neighbours']
    neighb_tensor_coords={}
    neighb_tensor_coords['month']=neighb_month_ids
    neighb_tensor_coords['country']=neighb_country_ids
    neighb_tensor_coords['neighbours']=neighb_country_ids
    
    
    # build dicts to tranform between month and index and country and index in the input
    # dataframe
    
    month_to_index={}
    for imonth,month in enumerate(neighb_month_ids):
        month_to_index[month]=imonth
        
    country_to_index={}
    for icountry,country in enumerate(neighb_country_ids):
        country_to_index[country]=icountry

    neighbours_table=df_neighbs.to_numpy()

    # assemble tensor

    for i in range(np.shape(neighbours_table)[0]):
        month=neighbours_table[i][0]
        imonth=month_to_index[month]
        
        country=neighbours_table[i][1]
        icountry=country_to_index[country]
        
        neighb=neighbours_table[i][2]
        ineighb=country_to_index[neighb]
                 
        neighb_data[imonth,icountry,ineighb]=1

    neighb_tensor=xarray.DataArray(data=neighb_data, coords=neighb_tensor_coords, dims=neighb_tensor_dims)

    return neighb_tensor
        
def _get_country_distances(country_ids,country_id_to_index):

    '''
    
    get_distances
    
    Fetched distances between countries using the Country class and stores them in a
    country x country tensor.
    
    '''
    
    ncountries=len(country_ids)
    
    distances=np.zeros((ncountries,ncountries))

    for country_idi in country_ids:
        country_indexi=country_id_to_index[country_idi]
            
        countryi=Country(country_idi)
                          
        lati=countryi.lat
        longi=countryi.lon
                       
        for country_idj in country_ids[country_indexi:]:
            country_indexj=country_id_to_index[country_idj]
            
            countryj=Country(country_idj)
                
            latj=countryj.lat
            longj=countryj.lon
                
            distances[country_indexi,country_indexj]= \
            distances[country_indexj,country_indexi]= \
            np.sqrt((lati-latj)**2.+(longi-longj)**2.)
        
    return distances    
    