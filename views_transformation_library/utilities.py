import numpy as np
import xarray
import functools
import ingester3
from ingester3.Country import Country
from ingester3.scratch import fetch_data

    
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

    df_neighbs = df_neighbs[columns].sort_values(
                                                 ["month_id", "a_id", "b_id"],
                                                 ascending=(True, True, True)
                                                 )

    neighb_month_ids = np.sort(df_neighbs['month_id'].unique())

    neighb_country_ids = np.sort(df_neighbs['b_id'].unique())

    nmonths = len(neighb_month_ids)
    
    ncountries = len(neighb_country_ids)

    neighb_data = np.zeros((nmonths, ncountries, ncountries))
    
    neighb_tensor_dims = ['month', 'country', 'neighbours']
    neighb_tensor_coords={}
    neighb_tensor_coords['month'] = neighb_month_ids
    neighb_tensor_coords['country'] = neighb_country_ids
    neighb_tensor_coords['neighbours'] = neighb_country_ids
    
    
    # build dicts to tranform between month and index and country and index in the input
    # dataframe
    
    month_to_index = {}
    for imonth, month in enumerate(neighb_month_ids):
        month_to_index[month] = imonth
        
    country_to_index = {}
    for icountry, country in enumerate(neighb_country_ids):
        country_to_index[country] = icountry

    neighbours_table = df_neighbs.to_numpy()

    # assemble tensor

    for i in range(np.shape(neighbours_table)[0]):
        month = neighbours_table[i][0]
        imonth = month_to_index[month]
        
        country = neighbours_table[i][1]
        icountry = country_to_index[country]
        
        neighb = neighbours_table[i][2]
        ineighb = country_to_index[neighb]
                 
        neighb_data[imonth, icountry, ineighb] = 1

    neighb_tensor = xarray.DataArray(data=neighb_data, coords=neighb_tensor_coords, dims=neighb_tensor_dims)

    return neighb_tensor


def get_country_distances(country_ids, country_id_to_index):

    """
    
    get_distances
    
    Fetched distances between countries using the Country class and stores them in a
    country x country tensor.
    
    """
    
    ncountries = len(country_ids)
    
    distances = np.zeros((ncountries, ncountries))

    for country_idi in country_ids:
        country_indexi = country_id_to_index[country_idi]
            
        countryi = Country(country_idi)
                          
        lati = countryi.lat
        longi = countryi.lon
                       
        for country_idj in country_ids[country_indexi:]:
            country_indexj = country_id_to_index[country_idj]
            
            countryj = Country(country_idj)
                
            latj = countryj.lat
            longj = countryj.lon
                
            distances[country_indexi, country_indexj] = \
            distances[country_indexj, country_indexi] = \
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
