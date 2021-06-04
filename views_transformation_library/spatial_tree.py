import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import time

def get_tree_lag(df,thetacrit,dfunction_option):
    '''
    get_tree_lag

    Driver function for computing tree-lagged features

    Arguments:

    df: dataframe containing one or more features to be lagged

    thetacrit: opening angle used to decide whether to open nodes - large values cause more
    aggressive aggregation of nodes

    dfunctions: a dict of function_name:function pairs, where the functions must take a single
    argument (distance between nodes) and return a single result (weight corresponding to distance)

    '''

    df=df.fillna(0.0)
    if not df.index.is_monotonic:
        df=df.sort_index()

    dfunctions=get_dfunctions(dfunction_option)

    tree=SpatialTree()

    tree.map_pgids(df)

    tree.build_tree()

    tree.stock(df)

    tree.walk(thetacrit,dfunctions)

    df_treelags=tree.tree_lag()

    return df_treelags


def get_dfunctions(dfunction_option):

    dfunctions={}

    if dfunction_option==0:

        def logd(d):
            return 1./np.log(1.+d)

        dfunctions['logd']=logd

    elif dfunction_option==1:

        def d_1(d):
            return 1./d

        dfunctions['d_1']=d_1

    elif dfunction_option==2:

        def d_2(d):
            return 1./d/d

        dfunctions['d_2']=d_2

    else:
        raise Exception(
                "Unrecognised distance function in tree. "
                f"Allowed: 0=1/ln(1+d), 1=1/d, 2=1/d^2, Supplied: {dfunction_option}"
            )

    return dfunctions


def get_grid_lag(df,threshold,dfunctions,split_criterion,keep_grids):

    '''
    get_grid_lag

    Driver function for computing adaptive-grid-lagged features

    Arguments:

    df: dataframe containing one or more features to be lagged

    threshold: number to be used to decide whether to split nodes

    dfunctions: a dict of function_name:function pairs, where the functions must take a single
    argument (distance between nodes) and return a single result (weight corresponding to distance)

    split_criterion: list containing one or more of the following strings:

    - 'min': nodes are split if their value of the feature exceeds the value of threshold

    keep_grids: flag instructing function to return the nodeids in the adaptive
    mesh at all timesteps

    '''

    tree=SpatialTree()

    tree.map_pgids(df)

    tree.build_tree()

    tree.stock(df)

    df_gridlags=tree.grid_lag(dfunctions,split_criterion,threshold,keep_grids)

    return df_gridlags

class SpatialNode():
    """
    SpatialNode() class
    Defines nodes for insertion in SpatialTree() class

    Nodes have:
    - nodeid: a unique id
    - masterindex, levelindex, level: book-keeping indices giving their tree level and
      location on that level (not needed once tree has been built)
    - parent: parent node
    - left, right, bottom, top: node boundaries
    - centre: node's geometric centre
    - isleaf: a flag indicating whether a leaf node
    - pgid: the corresponding priogrid cell, if a leaf node (-1 if not a leaf node)
    - nleaf: the number of leaf nodes contained within the node's boundaries
    - children: list of node's children
    - features: dict of features loaded into the node by the .stock() method

    """
    def __init__(self):
        self.nodeid = -1
        self.masterindex = -1
        self.levelindex = -1
        self.level = -1
        self.parent = -1
        self.left = -1
        self.right = -1
        self.top = -1
        self.bottom = -1
        self.centre = np.array([-1.0,-1.0])
        self.isleaf = False
        self.pgid = -1
        self.nleaf = 0
        self.children = []
        self.features = {}

    def __repr__(self):
        return f""" nodeid: {self.nodeid}\n masterindex: {self.masterindex}\n levelindex: {self.levelindex}
        level: {self.level}\n parent: {self.parent}\n left: {self.left}
        right: {self.right}\n bottom: {self.bottom}\n top: {self.top}
        centre: {self.centre}\n isleaf: {self.isleaf}\n children: {self.children}"""

class SpatialTree():
    '''
    SpatialTree() class
    Defines tree class and methods for building, walking and stocking

    '''

    def __init__(self):
        self.ncells=0
        self.power=0
        self.pgids=None
        self.pgid_to_longlat={}
        self.longlat_to_pgid={}
        self.pgid_to_index={}
        self.nodes=[]
        self.interaction_lists=None
        self.interaction_weights=None
        self.times=None
        self.features=None
        self.weightfunctions=None

    def map_pgids(
        self,
		df
	):
        '''
	    map_pgids
	
	    This function builds a 2D map in longitude-latitude from the pgids contained in
	    the input dataframe, and creates dicts allowing quick transformation from (long,lat)
	    to pgid and vice versa.
	
	    The pgids are embedded and centred in the smallest possible square grid whose side
	    is an integer power of 2
	    '''
	
        PG_STRIDE=720
	
		# get unique pgids
	
        pgids=np.array(list({idx[1] for idx in df.index.values}))
        pgids=np.sort(pgids)
        self.pgids=pgids
		
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

        self.ncells=ncells
        self.power=power

        # centre the pgids

        inudgelong=int((ncells-longmax)/2)
        inudgelat=int((ncells-latmax)/2)

        longitudes+=inudgelong
        latitudes+=inudgelat

		# make dicts to transform between pgids and (long,lat) coordinate

        for i,pgid in enumerate(pgids):
            self.pgid_to_longlat[pgid]=(longitudes[i],latitudes[i])
            self.longlat_to_pgid[(longitudes[i],latitudes[i])]=pgid
            self.pgid_to_index[pgid]=i
        return
		
    def build_tree(self):

        '''
	    build_tree
	
	    This function builds the tree from the leaf nodes downwards.
	
	    A list with space for all possible nodes is first built and populated with the
	    leaf nodes actually present in the df the tree was initialised with.
	
	    masterindex is a book-keeping device which is a unique location in the list of
	    all possible nodes.
	
	    levelindex is retained for debugging purposes - each node has a number indicating
	    its position in its level, constructed in the same way one would number the
	    squares on a chessboard, stating with a1, b1, c1, etc.
	
	    Once leaf nodes are populated, work down through the levels, using nodes in the
	    level above to generate parent nodes in the current level, using masterindex to
	    avoid creating the same parent twice.
	
	    Once this is done, the list of all possible nodes is compactified to discard
	    un-needed indices, and all nodeids are reassigned so that the final list of nodes
	    is contiguous.
	
	    '''
	
        powers=[((2**p)**2) for p in range(self.power+1)]

        cupowers=np.cumsum(powers)

        nodes_provisional=[None for i in range(cupowers[-1])]

        # populate leaf nodes

        for pgid in self.pgids:
            ix,iy=self.pgid_to_longlat[pgid]

            levelindex=ix+iy*self.ncells
            masterindex=cupowers[-2]+levelindex+1

            if nodes_provisional[masterindex] is None:
                node=SpatialNode()
                node.level=self.power
                node.levelindex=levelindex
                node.masterindex=masterindex
                node.left=ix
                node.right=ix+1
                node.bottom=iy
                node.top=iy+1
                node.centre=np.array([ix+0.5,iy+0.5])
                node.isleaf=True
                node.nleaf=1
                node.pgid=pgid

                nodes_provisional[masterindex]=node

        # populate lower levels

        for p in range(self.power-1,-1,-1):
            istart=cupowers[p]-1
            iend=cupowers[p+1]
            ncellsp=2**p
            nodesize=self.ncells/ncellsp
            for inode in range(istart,iend):
                if nodes_provisional[inode] is not None:
                    node=nodes_provisional[inode]
                    ix=int(node.centre[0]/float(self.ncells)*float(ncellsp))
                    iy=int(node.centre[1]/float(self.ncells)*float(ncellsp))
                    levelindex=ix+iy*ncellsp
                    if (p-1)<0:
                        masterindex=0
                    else:
                        masterindex=cupowers[p-1]+levelindex

                    if nodes_provisional[masterindex] is None:
                        parent=SpatialNode()
                        parent.level=p
                        parent.levelindex=levelindex
                        parent.masterindex=masterindex
                        parent.left=ix*nodesize
                        parent.right=parent.left+nodesize
                        parent.bottom=iy*nodesize
                        parent.top=parent.bottom+nodesize
                        parent.centre=np.array([(parent.left+parent.right)/2,(parent.bottom+parent.top)/2])
                        parent.children.append(node.masterindex)
                        parent.nleaf+=node.nleaf
                        nodes_provisional[masterindex]=parent
                    else:
                        parent=nodes_provisional[masterindex]
                        parent.children.append(node.masterindex)
                        parent.nleaf+=node.nleaf

                    node.parent=parent.masterindex

        # compactify list of nodes, and reassign unique indices

        provisional_to_final={}

        for node in nodes_provisional:
            if node is not None:
                self.nodes.append(node)
                node.nodeid=len(self.nodes)-1
                provisional_to_final[node.masterindex]=node.nodeid

        for node in self.nodes:
            if node.parent!=-1:
                node.parent=provisional_to_final[node.parent]
            node.children=[provisional_to_final[child] for child in node.children]

    def walk(self,thetacrit,dfunctions):

        '''
	    walk
	
	    This function generates the list of nodes any given node will import data from,
	    when one lags variables using the tree, as well as weights based on the distance
	    between nodes' centres.
	
	    The arguments are
	
	    - thetacrit: angle used to decide if a candidate node should be added to a target
	      node's interaction list, based on the size of the candidate node and
	      the distance between the candidate node and the target node
	
	    - dfunctions: dict of named functions of internode distance d only, used to
	      compute weights of node-node interactions
	    '''

        interaction_lists={}
        interaction_weights={}
        self.weightfunctions=dfunctions

        for targetnode in self.nodes:
            if not(targetnode.isleaf):
                continue
            targetpos=targetnode.centre
            targetid=targetnode.pgid
            interaction_list=[]
            interaction_weight={}
            nodestodo=[0,]

            while (len(nodestodo)>0):
                inode=nodestodo[0]
                node=self.nodes[inode]

                d=targetpos-node.centre+0.01
                d=np.sqrt(np.dot(d,d))
                h=node.top-node.bottom
                theta=h/d
                if theta>thetacrit:
                # attempt to split node
                    if node.isleaf:
                    # if node is a leaf node, it cannot be split: add to interaction list
                        if not(targetnode.nodeid == node.nodeid):
                            interaction_list.append(node.nodeid)
                            weights={}
                            for dfunction in dfunctions.keys():
                                weights[dfunction]=dfunctions[dfunction](d)
                            interaction_weight[node.nodeid]=weights

                        nodestodo.pop(0)
                    else:
                    # otherwise get children and put in list of nodes to check
                        for ichild in node.children:
                            nodestodo.append(ichild)

                        nodestodo.pop(0)
                else:
                # add to interaction list
                    if node.nleaf>0:
                    # TODO not sure this check is still necessary
                        interaction_list.append(node.nodeid)
                        weights={}
                        for dfunction in dfunctions.keys():
                            weights[dfunction]=dfunctions[dfunction](d)
                        interaction_weight[node.nodeid]=weights

                    nodestodo.pop(0)

            interaction_lists[targetid]=interaction_list
            interaction_weights[targetid]=interaction_weight

        self.interaction_lists=interaction_lists
        self.interaction_weights=interaction_weights

        return


    def df_to_tensor_no_strides(self,df):

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

    def df_to_tensor_strides(self,
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

    def stock(self,df,use_stride_tricks=True):

        '''
        stock

        This function loads features into the tree.

        Data from the input df is first cast to a 3D tensor with dims time x pgid x
        features.

        The tensor is used to populate the leaf nodes, and values then just propagate
        down through the tree via the parents.

        At present, values of parents are the *sums* of those of their children.

        '''

        features=df.columns
		
        self.features=features

        times=np.array(list({idx[0] for idx in df.index.values}))
        times=list(np.sort(times))
        self.times=times

        for node in self.nodes:
            for feature in features:
                node.features[feature]=np.zeros(len(times))

                # TODO weighted centres

        if (use_stride_tricks):
            tensor3d=self.df_to_tensor_strides(df)
        else:
            tensor3d=self.df_to_tensor_no_strides(df)

        for ifeature,feature in enumerate(features):
            for node in self.nodes:
                if node.isleaf:
                    pgid=node.pgid
                    ipgid=self.pgid_to_index[pgid]
                    vals=tensor3d[:,ipgid,ifeature]

                    node.features[feature]+=vals
                    while node.parent>=0:
                        node=self.nodes[node.parent]

                        node.features[feature]+=vals

        return

    def tree_lag(self):

        '''
        tree_lag

        This function computes features spatially lagged using the tree for every pgid.

        The interaction lists and weights, and the features loaded into each node are
        combined in weighted sums.

        The lagged features are then packed into a dataframe.

        '''

        dim0=len(self.times)
        dim1=len(self.pgids)
        dim2=len(self.features)
        nweightfunctions=len(self.weightfunctions)
        weightkeys=list((self.weightfunctions.keys()))

        treelags=np.empty((dim0,dim1,dim2*nweightfunctions))

        for ifeature,feature in enumerate(self.features):
            for node in self.nodes:
                if node.isleaf:
                    pgid=node.pgid
                    ipgid=self.pgid_to_index[pgid]

                    interactions=self.interaction_lists[pgid]
                    for iweight,weightkey in enumerate(weightkeys):
                        weights=np.array([self.interaction_weights[pgid][partner][weightkey] for partner in interactions])

                        sums=np.zeros(len(self.times))

                        for partner,weight in zip(interactions,weights):
                            node=self.nodes[partner]
                            sums+=node.features[feature]*weight
                        treelags[:,ipgid,ifeature*nweightfunctions+iweight]=sums

        # create tensor to pack into df

        flat=np.empty((dim0*dim1,dim2*nweightfunctions))

        for ichunk in range(dim0):
            flat[ichunk*dim1:(ichunk+1)*dim1,:]=treelags[ichunk,:,:]

        df_column_names=['treelag_'+weight+'_'+feature for feature in self.features for weight in self.weightfunctions.keys()]

        df_index=pd.MultiIndex.from_product([self.times, self.pgids],names=['month', 'pgid'])

        df_treelags=pd.DataFrame(flat, index=df_index, columns=df_column_names)

        return df_treelags


    def grid_lag(self,dfunctions,split_criterion='min',threshold=10000,keep_grids=False):

        '''
        grid_lag

        This function uses the tree in a different way to build, at each timestep, a single
        grid from the tree nodes, with nodes being split according to a refinement
        criterion. The available criteria are:

        - 'min': starting with the root node, nodes whose value at a given timestep is
          greater than the threshold are recursively split

        '''


        if (keep_grids):
            grids=[]

        dim0=len(self.times)
        dim1=len(self.pgids)
        dim2=len(self.features)

        self.weightfunctions=dfunctions

        nweightfunctions=len(self.weightfunctions)

        weightkeys=list((self.weightfunctions.keys()))

        gridlags=np.empty((dim0,dim1,dim2*nweightfunctions))


        def split_min(threshold,time,feature):
            grid_nodes=[]
            nodestodo=[0,]

            while len(nodestodo)>0:

                inode=nodestodo.pop(0)
                node=self.nodes[inode]
                val=node.features[feature][time]

                if val>threshold:
                    if node.isleaf:

                        grid_nodes.append(inode)
                    else:
                        for child in node.children:

                            nodestodo.append(child)

                else:
                    if node.nleaf>0:

                        grid_nodes.append(inode)

            return grid_nodes

        if split_criterion=='min':
            split_decider=split_min

        else:
            print('Unknown criterion for splitting nodes: ',split_criterion)
            raise Exception()

        for itime,time in enumerate(self.times):
            for ifeature,feature in enumerate(self.features):

                grid_nodes=split_decider(threshold,itime,feature)

                if(keep_grids):
                    grids.append(grid_nodes)

                for targetnode in self.nodes:
                    if not(targetnode.isleaf):
                        continue
                    targetpos=targetnode.centre

                    targetnodeid=targetnode.nodeid

                    sums=np.zeros((nweightfunctions))

                    for inode in grid_nodes:
                        node=self.nodes[inode]

                        if node.nodeid == targetnodeid:
                            continue

                        val=node.features[feature][itime]

#                        d=np.array(targetpos)-np.array(node.centre)+0.01
                        d=targetpos-node.centre
                        d=np.sqrt(np.dot(d,d))

                        for iweight,weightkey in enumerate(weightkeys):
                            sums[iweight]+=val*dfunctions[weightkey](d)

                    ipgid=self.pgid_to_index[targetnode.pgid]
                    gridlags[itime,ipgid,ifeature*nweightfunctions:(ifeature+1)*nweightfunctions]=sums

        flat=np.empty((dim0*dim1,dim2*nweightfunctions))

        for ichunk in range(dim0):
            flat[ichunk*dim1:(ichunk+1)*dim1,:]=gridlags[ichunk,:,:]

        df_column_names=['gridlag_'+weight+'_'+feature for feature in self.features for weight in self.weightfunctions.keys()]

        df_index=pd.MultiIndex.from_product([self.times, self.pgids],names=['month', 'pgid'])

        df_gridlags=pd.DataFrame(flat, index=df_index, columns=df_column_names)

        if keep_grids:
            return df_gridlags,grids
        else:
            return df_gridlags





