""" Spatial transforms

"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from views_transformation_library import utilities


def sptdist(tensor, index, return_values='distances', k=1, nu=1.0, power=0.0, last_month = 0):
    """
    sptdist

    For every point in the supplied tensor, uses scipy.spatial cKDTree to find the nearest k
    past events (where an event is any non-zero value in the input df) and returns either
    the mean spacetime distance to the k events, or the mean of
    (size of event)/(spacetime distance)**power.

    Arguments:

    tensor:        input tensor
    return_values: choice of what to return. Allowed values:

                   distances - return mean spacetime distance to nearest k events

                   weights   - return mean of (size of event)/(spacetime distance)**power

    k:             number of nearest events to be averaged over

    nu:            weighting to be applied to time-component of distances, so that
                   spacetime distance = sqrt(delta_latitude^2 + delta_longitude^2 +
                   nu^2*delta_t^2)

    power:         power to which distance is raised when computing weights. Negative
                   values are automatically converted to positive values.

    """

    def get_space_time_distances(
        tensor,
        times,
        pgids,
        pgid_to_longlat,
        time_to_index,
        return_values,
        k,
        nu,
        power,
        last_month
    ):
        """
        space_time_distances

        Finds spacetime distances, scaling spatial distances to degrees (for continuity with
        previous versions of this function) and stretching the time axis by nu.

        If k>1, averaging is performed.

        """

        PGID_TO_DEGREES = 0.5

        output = np.zeros((len(times), len(pgids), tensor.shape[-1]))

        for ifeature in range(tensor.shape[-1]):

            events3d = utilities.build_4d_tensor(tensor[:, :, ifeature].reshape(len(times), len(pgids), 1),
                                                 pgids, pgid_to_index, times, time_to_index, ncells, ncells,
                                                 pgid_to_longlat).reshape(ncells, ncells, len(times))

            for time in times[:last_month]:

                tindex = time_to_index[time]

                points = np.array(np.where(events3d[:, :, :tindex + 1] > 0)).astype(float).T

                if len(points) == 0:
                    output[tindex,:,ifeature] = 999.
                else:
                    points[:, 0] *= PGID_TO_DEGREES
                    points[:, 1] *= PGID_TO_DEGREES
                    points[:, 2] *= nu
                    btree = cKDTree(data=points, leafsize=20)

                    feature = 999.

                    for ipgid, pgid in enumerate(pgids):
                        ilong, ilat = pgid_to_longlat[pgid]

                        sptime_dists, ipoints = btree.query([ilong * PGID_TO_DEGREES, ilat * PGID_TO_DEGREES,
                                                                 nu * tindex], k)

                        if k == 1:
                            ipoints = [ipoints, ]
                            sptime_dists = [sptime_dists, ]

                        if return_values == 'distances':

                            if k == 1:
                                feature = sptime_dists[0]
                            else:
                                feature = np.mean(sptime_dists)

                        elif return_values == 'weights':

                            featurei = np.zeros(k)

                            for i, ipoint in enumerate(ipoints):

                                coords = points[ipoint]

                                ilongp = int(coords[0] / PGID_TO_DEGREES)
                                ilatp = int(coords[1] / PGID_TO_DEGREES)
                                itimep = int(coords[2] / nu)

                                if sptime_dists[i] == 0.0:
                                    featurei[i] = tensor[ilongp, ilatp, itimep, 0]
                                else:
                                    featurei[i] = tensor[ilongp, ilatp, itimep, 0] / (sptime_dists[i] ** power)

                                feature = np.mean(featurei)

                        output[tindex, ipgid, ifeature] = feature

        return output

    if return_values not in ['distances', 'weights']:
        raise Exception("unknown return_values; ", return_values,
                        " - allowed choices: distances weights")

    power = np.abs(power)

    times, time_to_index, index_to_time = utilities.map_times(index)

    pgids, pgid_to_longlat, longlat_to_pgid, pgid_to_index, index_to_pgid, ncells, pwr = \
        utilities.map_pgids_2d(index)

    tensor = get_space_time_distances(tensor,times,pgids,pgid_to_longlat,time_to_index,return_values,
        k,nu,power,last_month)

    return tensor

def sp_tree(tensor, index, thetacrit, dfunction_option):
    """
    get_tree_lag

    Driver function for computing tree-lagged features

    Arguments:

    df: dataframe containing one or more features to be lagged

    thetacrit: opening angle used to decide whether to open nodes - large values cause more
    aggressive aggregation of nodes

    dfunction_option: an integer selecting which distance weighting to use:

                     - 0: ln(1+d) weighting

                     - 1: 1/d weighting

                     - 2: 1/d^2 weighting

    """

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
            self.centre = np.array([-1.0, -1.0])
            self.isleaf = False
            self.pgid = -1
            self.nleaf = 0
            self.children = []
            self.features = []

        def __repr__(self):
            return f""" nodeid: {self.nodeid}\n masterindex: {self.masterindex}\n levelindex: {self.levelindex}
            level: {self.level}\n parent: {self.parent}\n left: {self.left}
            right: {self.right}\n bottom: {self.bottom}\n top: {self.top}
            centre: {self.centre}\n isleaf: {self.isleaf}\n children: {self.children}"""

    class SpatialTree():
        """
        SpatialTree() class
        Defines tree class and methods for building, walking and stocking

        """

        def __init__(self):
            self.ncells = 0
            self.power = 0
            self.pgids = None
            self.pgid_to_longlat = {}
            self.longlat_to_pgid = {}
            self.pgid_to_index = {}
            self.index_to_pgid = {}
            self.nodes = []
            self.interaction_lists = None
            self.interaction_weights = None
            self.times = None
            self.features = None
            self.weightfunctions = None

        def build_tree(self, index):

            """
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

            """

            self.pgids, self.pgid_to_longlat, self.longlat_to_pgid, self.pgid_to_index, \
                self.index_to_pgid, self.ncells, self.power = utilities.map_pgids_2d(index)

            powers = [((2 ** p) ** 2) for p in range(self.power + 1)]

            cupowers = np.cumsum(powers)

            nodes_provisional = [None for i in range(cupowers[-1])]

            # populate leaf nodes

            for pgid in self.pgids:
                ix, iy = self.pgid_to_longlat[pgid]

                levelindex = ix + iy * self.ncells
                masterindex = cupowers[-2] + levelindex + 1

                if nodes_provisional[masterindex] is None:
                    node = SpatialNode()
                    node.level = self.power
                    node.levelindex = levelindex
                    node.masterindex = masterindex
                    node.left = ix
                    node.right = ix + 1
                    node.bottom = iy
                    node.top = iy + 1
                    node.centre = np.array([ix + 0.5, iy + 0.5])
                    node.isleaf = True
                    node.nleaf = 1
                    node.pgid = pgid

                    nodes_provisional[masterindex] = node

            # populate lower levels

            for p in range(self.power - 1, -1, -1):
                istart = cupowers[p] - 1
                iend = cupowers[p + 1]
                ncellsp = 2 ** p
                nodesize = self.ncells / ncellsp
                for inode in range(istart, iend):
                    if nodes_provisional[inode] is not None:
                        node = nodes_provisional[inode]
                        ix = int(node.centre[0] / float(self.ncells) * float(ncellsp))
                        iy = int(node.centre[1] / float(self.ncells) * float(ncellsp))
                        levelindex = ix + iy * ncellsp
                        if (p - 1) < 0:
                            masterindex = 0
                        else:
                            masterindex = cupowers[p - 1] + levelindex

                        if nodes_provisional[masterindex] is None:
                            parent = SpatialNode()
                            parent.level = p
                            parent.levelindex = levelindex
                            parent.masterindex = masterindex
                            parent.left = ix * nodesize
                            parent.right = parent.left + nodesize
                            parent.bottom = iy * nodesize
                            parent.top = parent.bottom + nodesize
                            parent.centre = np.array(
                                [(parent.left + parent.right) / 2, (parent.bottom + parent.top) / 2])
                            parent.children.append(node.masterindex)
                            parent.nleaf += node.nleaf
                            nodes_provisional[masterindex] = parent
                        else:
                            parent = nodes_provisional[masterindex]
                            parent.children.append(node.masterindex)
                            parent.nleaf += node.nleaf

                        node.parent = parent.masterindex

            # compactify list of nodes, and reassign unique indices

            provisional_to_final = {}

            for node in nodes_provisional:
                if node is not None:
                    self.nodes.append(node)
                    node.nodeid = len(self.nodes) - 1
                    provisional_to_final[node.masterindex] = node.nodeid

            for node in self.nodes:
                if node.parent != -1:
                    node.parent = provisional_to_final[node.parent]
                node.children = [provisional_to_final[child] for child in node.children]

        def walk(self, thetacrit, dfunctions):

            """
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
            """

            interaction_lists = {}
            interaction_weights = {}
            self.weightfunctions = dfunctions

            for targetnode in self.nodes:
                if not targetnode.isleaf:
                    continue
                targetpos = targetnode.centre
                targetid = targetnode.pgid
                interaction_list = []
                interaction_weight = {}
                nodestodo = [0, ]

                while len(nodestodo) > 0:
                    inode = nodestodo[0]
                    node = self.nodes[inode]

                    d = targetpos - node.centre + 0.01
                    d = np.sqrt(np.dot(d, d))
                    h = node.top - node.bottom
                    theta = h / d
                    if theta > thetacrit:
                        # attempt to split node
                        if node.isleaf:
                            # if node is a leaf node, it cannot be split: add to interaction list
                            if not (targetnode.nodeid == node.nodeid):
                                interaction_list.append(node.nodeid)
                                weights = {}
                                for dfunction in dfunctions.keys():
                                    weights[dfunction] = dfunctions[dfunction](d)
                                interaction_weight[node.nodeid] = weights

                            nodestodo.pop(0)
                        else:
                            # otherwise get children and put in list of nodes to check
                            for ichild in node.children:
                                nodestodo.append(ichild)

                            nodestodo.pop(0)
                    else:
                        # add to interaction list
                        if node.nleaf > 0:
                            # TODO not sure this check is still necessary
                            interaction_list.append(node.nodeid)
                            weights = {}
                            for dfunction in dfunctions.keys():
                                weights[dfunction] = dfunctions[dfunction](d)
                            interaction_weight[node.nodeid] = weights

                        nodestodo.pop(0)

                interaction_lists[targetid] = interaction_list
                interaction_weights[targetid] = interaction_weight

            self.interaction_lists = interaction_lists
            self.interaction_weights = interaction_weights

            return

        def stock(self, tensor3d, index):

            """
            stock

            This function loads features into the tree.

            The input tensor is used to populate the leaf nodes, and values then just propagate
            down through the tree via the parents.

            At present, values of parents are the *sums* of those of their children.

            """

            self.features = np.arange(tensor3d.shape[-1])

            self.times, _, _ = utilities.map_times(index)

            for node in self.nodes:
                for feature in self.features:
                    node.features.append(np.zeros(len(self.times)))

                    # TODO weighted centres

            for ifeature, feature in enumerate(self.features):
                for node in self.nodes:
                    if node.isleaf:
                        pgid = node.pgid
                        ipgid = self.pgid_to_index[pgid]
                        vals = tensor3d[:, ipgid, ifeature]

                        node.features[feature] += vals
                        while node.parent >= 0:
                            node = self.nodes[node.parent]

                            node.features[feature] += vals

            return

        def tree_lag(self):

            """
            tree_lag

            This function computes features spatially lagged using the tree for every pgid.

            The interaction lists and weights, and the features loaded into each node are
            combined in weighted sums.

            The lagged features are then packed into a tensor.

            """

            dim0 = len(self.times)
            dim1 = len(self.pgids)
            dim2 = len(self.features)
            nweightfunctions = len(self.weightfunctions)
            weightkeys = list((self.weightfunctions.keys()))

            treelags = np.zeros((dim0, dim1, dim2 * nweightfunctions),dtype=tensor.dtype)

            for ifeature, feature in enumerate(self.features):
                for node in self.nodes:
                    if node.isleaf:
                        pgid = node.pgid
                        ipgid = self.pgid_to_index[pgid]

                        interactions = self.interaction_lists[pgid]
                        for iweight, weightkey in enumerate(weightkeys):
                            weights = np.array([self.interaction_weights[pgid][partner][weightkey] for partner in
                                                interactions])

                            sums = np.zeros(len(self.times))

                            for partner, weight in zip(interactions, weights):
                                node = self.nodes[partner]
                                sums += node.features[feature] * weight
                            treelags[:, ipgid, ifeature * nweightfunctions + iweight] = sums

            return treelags

    def get_dfunctions(dfunction_option):

        dfunctions = {}

        match (dfunction_option):
            case (0):

                def logd(d):
                    return 1. / np.log(1. + d)

                dfunctions[dfunction_option] = logd

            case (1):

                def d_1(d):
                    return 1. / d

                dfunctions[dfunction_option] = d_1

            case (2):

                def d_2(d):
                    return 1. / d / d

                dfunctions[dfunction_option] = d_2

            case _:
                raise Exception(
                    "Unrecognised distance function in tree. "
                    f"Allowed: 0=1/ln(1+d), 1=1/d, 2=1/d^2, Supplied: {dfunction_option}"
                )

        return dfunctions

    tensor = np.where(np.isnan(tensor),0.0,tensor)

    dfunctions = get_dfunctions(dfunction_option)

    tree = SpatialTree()

    tree.build_tree(index)

    tree.stock(tensor,index)

    tree.walk(thetacrit, dfunctions)

    return tree.tree_lag()


def splag4d(tensor,index,kernel_inner=1, kernel_width=1,
                kernel_power=0, norm_kernel=0):
    """
    splag4d created 19/03/2021 by Jim Dale

    Performs spatial lags on a dataframe by transforming from flat format to 4d tensor
    with dimensions longitude x latitude x time x features.

    Spatial lagging can then be done as a 2d convolution on long-lat slices using
    scipy convolution algorithms.

    Arguments:

    df:                a dataframe of series to be splagged

    use_stride_tricks: boolean, decide to use stride_tricks or not (optional,
                       defaults to True)

    kernel_inner:      inner border of convolution region (set to 1 to exclude central
                       cell)

    kernel_width:      width in cells of kernel, so outer radius of kernel =
                       kernel_inner + kernel_width

    kernel_power:      weight values of cells by (distance from centre of kernel)**
                       (-kernel_power) - set to zero for no distance weighting

    norm_kernel:       set to 1 to normalise kernel weights

    Returns:

                       A tensor of all lagged columns from input tensor

    """

    def build_kernel_weights(kernel_inner, kernel_width, kernel_power, norm_kernel):
        kernel_inner = int(kernel_inner)
        kernel_width = int(kernel_width)

        kernel_size = int(2 * (kernel_inner + kernel_width)) - 1
        weights = np.ones((kernel_size, kernel_size))

        kernel_centre = int((kernel_size + 1) / 2) - 1

        for ix in range(kernel_size):
            dx = ix - kernel_centre
            for iy in range(kernel_size):
                dy = iy - kernel_centre
                if (abs(dx) < kernel_inner) and (abs(dy) < kernel_inner):
                    weights[ix, iy] = 0.0
                else:
                    rxy = np.sqrt(dx * dx + dy * dy)
                    weights[ix, iy] /= rxy ** kernel_power

        if norm_kernel:
            weights /= np.sum(weights)

        return weights

    def get_splags(
            tensor,
            longrange,
            latrange,
            times,
            pgids,
            time_to_index,
            pgid_to_index,
            pgid_to_longlat,
            weights
    ):

        # use scipy convolution function to do 2d convolution on tensor slices

        for ifeature in range(tensor.shape[-1]):
            splag_feature = utilities.build_4d_tensor(tensor[:, :, ifeature].reshape(len(times), len(pgids), 1),
                                                 pgids, pgid_to_index, times, time_to_index, longrange, latrange,
                                                 pgid_to_longlat).reshape(ncells, ncells, len(times))

            for time in times:
                tindex = time_to_index[time]

                splag_feature[:,:,tindex] = ndimage.convolve(splag_feature[:,:,tindex], weights,
                                                             mode='constant', cval=0.0)

            for ipgid, pgid in enumerate(pgids):
                ilong, ilat = pgid_to_longlat[pgid]
                tensor[:,ipgid,ifeature] = splag_feature[ilong,ilat,:]

        return tensor

    tensor = np.where(np.isnan(tensor), 0.0, tensor)

    weights = build_kernel_weights(kernel_inner, kernel_width, kernel_power, norm_kernel)

    times, time_to_index, index_to_time = utilities.map_times(index)

    pgids, pgid_to_longlat, longlat_to_pgid, pgid_to_index, index_to_pgid, ncells, power = \
        utilities.map_pgids_2d(index)

    splags = get_splags(tensor, ncells, ncells, times,pgids, time_to_index, pgid_to_index,
                        pgid_to_longlat, weights)

    return splags
