import numpy as np
import xarray
import dill
import functools
import ingester3
from ingester3.Country import Country
from ingester3.scratch import fetch_data

class TensorContainer:

    def __init__(self,tensor,index,dne=-np.inf,missing=np.nan):
        self.tensor = tensor
        self.index = index
        self.dne = dne
        self.missing = missing

def df_to_tensor_no_strides(df):
    """
    df_to_tensor_no_strides created 13/03/2021 by Jim Dale
    Uses regular array indexing to create a tensorlike from a dataframe.

    dim0 of the tensor corresponds to level 0 of the df multiindex,
    dim1 corresponds to level 1 of the df multiindex, and dim2 corresponds to
    the df's columns.
    """

    # get shape of dataframe

    dim0,dim1=df.index.levshape

    dim2=df.shape[1]

    # check that df can in principle be tensorised

    if dim0*dim1!=df.shape[0]:
        raise Exception("df cannot be cast to a tensor - dim0 * dim1 != df.shape[0]",dim0,dim1,df.shape[0])


    flat=df.to_numpy()

    tensor3d=np.full((dim0,dim1,dim2),np.nan)
    chunksize=dim1
    nchunks=dim0
    for ichunk in range(nchunks):
        tensor3d[ichunk,:,:]=flat[ichunk*chunksize:(ichunk+1)*chunksize,:]

    return tensor3d 

def df_to_tensor_strides(df):
    """
    df_to_tensor created 13/03/2021 by Jim Dale
    Uses as_strided from numpy stride_tricks library to create a tensorlike
    from a dataframe.

    dim0 of the tensor corresponds to level 0 of the df multiindex,
    dim1 corresponds to level 1 of the df multiindex, and dim2 corresponds to
    the df's columns.
    """

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
    

def map_times(index):
    """
    
    map_times
    
    Gets the times from a dataframe, assuming that they are the first index, and returns
    a list of unique times, and dicts which convert between the actual time, and the
    index in the list.

    """
    # get unique times

    times=np.array(list({idx[0] for idx in index.values}))
    times=list(np.sort(times))
    
    # make dicts to transform between times and the index of a time in the list
    
    time_to_index={}
    index_to_time={}
    for i,time in enumerate(times):
        time_to_index[time]=i
        index_to_time[i]=time
    
    return times,time_to_index,index_to_time

    
def map_spaces(index):
    """
    
    map_spaces
    
    Gets the pgids from a dataframe, assuming that they are the second index, and 
    returns a 1d list of unique pgids, together with dicts which convert between 
    the actual pgid and the index in the list.
    
    """
    
    space_ids=np.array(list({idx[1] for idx in index.values}))

    space_ids=np.sort(space_ids)
    
    space_id_to_index={}
    index_to_space_id={}

    for i,space_id in enumerate(space_ids):
        space_id_to_index[space_id]=i
        index_to_space_id[i]=space_id

    return space_ids,space_id_to_index,index_to_space_id
    
    
def map_pgids_2d(index):
    """
	map_pgids_2d
	
	This function builds a 2D map in longitude-latitude from the pgids contained in
	the input dataframe assuming they are the second index column, and creates dicts 
	allowing quick transformation from (long,lat) to pgid and vice versa.
	
	The pgids are embedded and centred in the smallest possible square grid whose side
	is an integer power of 2

	"""
	
    PG_STRIDE=720

    # get unique pgids
	
    pgids=np.array(list({idx[1] for idx in index.values}))
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
    
def build_4d_tensor(
        tensor3d,
        pgids,
        pgid_to_index,
        times,
        time_to_index,
        longrange,
        latrange,
        pgid_to_longlat
):
    """

    build_4d_tensor

    This function accepts an input dataframe and the dicts generated by the map_times,
    map_features and map_pgids_2d functions, and returns a 4d latitude x longitude x
    time x feature tensor.


    """

    # convert 3d tensor into longitude x latitude x time x feature tensor

    tensor4d = np.zeros((longrange, latrange, len(times), tensor3d.shape[-1]))
    for pgid in pgids:

        pgindex = pgid_to_index[pgid]
        for time in times:
            tindex = time_to_index[time]
            ilong = pgid_to_longlat[pgid][0]
            ilat = pgid_to_longlat[pgid][1]

            tensor4d[ilong, ilat, tindex, :] = tensor3d[tindex, pgindex, :]

    return tensor4d
    
def tensor3d_to_tensor4d(tensor3d,index,use_stride_tricks):

    """

    df_to_datacube
    
    This function streamlines the pgid, time and feature mapping, together with the 4d
    tensor build, so that a df is accepted and a 4d tensor is produced to enable quick
    plotting.

    """

    pgids,pgid_to_longlat,longlat_to_pgid,pgid_to_index,index_to_pgid,ncells,power=_map_pgids_2d(index)

    times,time_to_index,index_to_time=_map_times(index)

    features=_map_features(index)
    
    tensor=build_4d_tensor(
                           tensor3d,
                           pgids,
                           pgid_to_index,
                           times,
                           time_to_index,
                           ncells,
                           ncells,
                           pgid_to_longlat,
                           )
    
    return tensor

def df_to_tensor4d(df):

    tensor3d = df_to_tensor_strides(df)

    pgids, pgid_to_longlat, longlat_to_pgid, pgid_to_index, index_to_pgid, ncells, power = map_pgids_2d(df.index)

    times, time_to_index, index_to_time = map_times(df.index)

    tensor4d = build_4d_tensor(
        tensor3d,
        pgids,
        pgid_to_index,
        times,
        time_to_index,
        ncells,
        ncells,
        pgid_to_longlat,
    )

    return tensor4d
    
def get_country_neighbours_tensor():

    """
    
    get_country_neighbours_tensor
    
    Fetches the country_country_month_expanded table and transforms it to a tensor with
    dimensions month x country x country, where (month_i, country_a, country_b)=1 
    indicates that country_a and country_b are neighbours in month_i. 
    
    Note that (month_i, country_a, country_a)=0, and that countries which never have any 
    neighbours do not appear in the tensor.
    
    """

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
        
def get_country_distances(country_ids,country_id_to_index):

    """
    
    get_distances
    
    Fetched distances between countries using the Country class and stores them in a
    country x country tensor.
    
    """
    
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


def dne_wrapper(fn):
    @functools.wraps(fn)
    def inner(tensor_container, *args, **kwargs):

        dne = tensor_container.dne
        missing = tensor_container.missing

        dne_mask = np.where(tensor_container.tensor == dne, True, False)

        tensor_container = fn(tensor_container, *args, **kwargs)

        tensor_container.tensor = np.where(np.logical_and(~dne_mask, tensor_container.tensor == dne),
                                           missing, tensor_container.tensor)

        tensor_container.tensor[dne_mask] = dne

        return tensor_container

    return inner
    
