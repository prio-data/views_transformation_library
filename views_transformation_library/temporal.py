""" Temporal transforms

"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from utilities import dne_wrapper

@dne_wrapper
def delta(tensor_container,time: int=1):

    missing = tensor_container.missing

    if time < 1:
        raise RuntimeError(f'Time below 1 passed to delta: {time}')

    for itime in reversed(range(tensor_container.tensor.shape[0])):
        if itime - time < 0:
            tensor_container.tensor[itime, :, :] = missing
        else:
            tensor_container.tensor[itime, :, :] = (tensor_container.tensor[itime, :, :] -
                                                    tensor_container.tensor[itime-time, :, :])

    return tensor_container

@dne_wrapper
def tlag(tensor_container, time: int):
    """
    tlag

    Shifts input series backwards in time

    Arguments:
        time: int specifying how many timesteps to shift backwards by

    """

    missing = tensor_container.missing

    if time < 1:
        raise RuntimeError(f'Time below 1 passed to tlag: {time}')

    for itime in reversed(range(tensor_container.tensor.shape[0])):
        if itime - time < 0:
            tensor_container.tensor[itime, :, :] = missing
        else:
            tensor_container.tensor[itime, :, :] = tensor_container.tensor[itime-time, :, :]

    return tensor_container

@dne_wrapper
def moving_average(tensor_container, time: int):
    """
    moving_average

    Computes moving average over a specified time window

    Arguments:
         window: integer size of moving time window over which to average

    """

    if time < 1:
        raise RuntimeError(f"Time below 1 passed to moving average: {time} \n")

    with warnings.catch_warnings(action="ignore"):

        tensor_container.tensor[time-1:,:,:] = np.nanmean(sliding_window_view(
                                               tensor_container.tensor, time, 0),axis=3)

        stub = np.zeros_like(tensor_container.tensor[:time-1,:,:])

        for itime in range(time-1):
            stub[itime, :, :] = np.nanmean(tensor_container.tensor[:itime+1,:,:], axis=0)

    tensor_container.tensor[:time-1,:,:] = stub

    return tensor_container

@dne_wrapper
def moving_sum(tensor_container, time: int):
    """
    moving_sum

    Computes moving sum over a specified time window

    Arguments:
         window: integer size of moving time window over which to average

    """

    if time < 1:
        raise RuntimeError(f"Time below 1 passed to moving sum: {time} \n")

    with warnings.catch_warnings(action="ignore"):

        tensor_container.tensor[time - 1:, :, :] = np.nansum(sliding_window_view(
                                                   tensor_container.tensor, time, 0),axis=3)

        stub = np.zeros_like(tensor_container.tensor[:time - 1, :, :])

        for itime in range(time-1):
            stub[itime, :, :] = np.nansum(tensor_container.tensor[:itime+1,:,:], axis=0)

    tensor_container.tensor[:time-1,:,:] = stub

    return tensor_container

@dne_wrapper
def cweq(tensor_container, value: float, seed=None):

    missing = tensor_container.missing

    for ifeature in range(tensor_container.tensor.shape[2]):
#        mask = np.where(tensor_container.tensor[:,:,ifeature] == value, 1, np.nan)
#        nan_mask = np.where(np.isnan(tensor_container.tensor[:,:,ifeature]), np.nan, 0)

#        nan_mask = np.where(nan_mask, np.nan, mask)

#        boolean_nan_mask = np.isnan(nan_mask)

        mask = np.where(tensor_container.tensor[:,:,ifeature] == value, 1, missing)
        nan_mask = np.where(tensor_container.tensor[:,:,ifeature]==missing, np.nan, 0)

        nan_mask = np.where(nan_mask, missing, mask)

        boolean_nan_mask = np.where(nan_mask==missing,True,False)

        cumsum_boolean = np.cumsum(~boolean_nan_mask, axis=0)

        for ispace in range(nan_mask.shape[1]):
            mask_diffs = np.diff(cumsum_boolean[:,ispace][boolean_nan_mask[:,ispace]],prepend=0)

            virow = nan_mask[:,ispace]
            nirow = boolean_nan_mask[:,ispace]

            virow[nirow] = -mask_diffs

            tensor_container.tensor[:,ispace, ifeature] = np.cumsum(virow)

            if seed is not None:
                seed1 = seed-1
                if tensor_container.tensor[:,ispace, ifeature][0] != 0:
                    for itime in range(tensor_container.tensor.shape[0]):
                        if tensor_container.tensor[itime,ispace,ifeature] == 0:
                            break
                        else:
                            tensor_container.tensor[itime, ispace, ifeature] += seed1


    return tensor_container

@dne_wrapper
def time_since(tensor_container, value=0, seed=None):
    """
    time_since

    time since event in series, where an event is where the series devaites from value.

    In order to compute a variable like "time since previous conflict
    event" we must apply a timelag to cweq() to get a series because
    for fitting a simultanous model we do not want the counter to be
    simultaneous to the event.

    Consider the data:

    event  : 0, 0, 1, 1, 0, 0 # Event
    cweq_0 : 1, 2, 0, 0, 1, 2 # count event while equals zero
    tisiev : ., 1, 2, 0, 0, 1 # time since event

    Fitting a model like "event ~ cweq0" makes no sense as cweq0 is
    always 0 if event=1.
    A model like "event ~ tsnp" makes more sense.
    We must apply a time lag to event before computing the counter to
    see how long time has elapsed since the previous event.

    Of course this isn't necessary for OSA modelling where all the
    rhs variables are time-lagged anyway but this is useful for
    dynamic simulation where X and predicted y are simulatenous.

    Arguments:
        value: float specifying value of series to follow
        seed: assumed time_since at beginning of series (defaults to None)

    """

    tensor_container = tlag(tensor_container,time=1)

    tensor_container = cweq(tensor_container, value=value, seed=seed)

    return tensor_container

@dne_wrapper
def decay(tensor_container, halflife: float):
    """
    decay

    Decay function, returning 2**(-s/halflife)

    See half-life formulation at
    https://en.wikipedia.org/wiki/Exponential_decay

    Arguments:
        halflife: float specifying time over which decay by a factor of 2 occurs

    """

    halflife = -halflife

    tensor_container.tensor = 2 ** (tensor_container.tensor/halflife)

    return tensor_container

@dne_wrapper
def onset_possible(tensor_container, window: int):
    """
    onset_possible

    Helper function which detects whether an onset (change from zero to non-zero state after at least window zero
    values) is possible. This function detects if no event occured in the preceeding window timesteps

    Arguments:
         window: integer specifying how many zero values must exist before a non-zero value to constitute an onset

    """

    tensor_container.tensor = (~rollmax(replace_na(tlag(tensor_container, 1),0.), window).
              astype(bool,copy=False)).astype(int,copy=False)

    return tensor_container

@dne_wrapper
def onset(tensor_container, window: int):
    """
    onset

    Computes onsets, where an onset occurs if, given the specified window, an onset is possible, and the value of s is
    non-zero

    Arguments:
         window: integer specifying how many zero values must exist before a non-zero value to constitute an onset

    """

    tensor = tensor_container.tensor

    tensor = (
        onset_possible(tensor.copy(), window).astype(bool) & tensor.copy().astype(bool)
                              ).astype(int)

    tensor_container.tensor = tensor

    return tensor_container

@dne_wrapper
def temporal_entropy(tensor_container,window,offset=0.):
    """
    temporal_entropy created 04/03/2022 by Jim Dale

    Computed entropy along the time axis within a window of length specified by 'window'.

    The entropy of a feature x over a window of length w is

    sum_(i=1,w) (x_i/X)log_2(x_i/X) where X = sum_(i=1,w) (x_i)

    Arguments:

    tensor:            a tensor of input data

    window:            integer size of window

    offset:            datasets containing mostly zeros will return
                       NaNs or Infs for entropy most or all of the time.
                       Since this is unlikely to be desirable, an
                       offset can be added to all feature values. so
                       that sensible values for entropy are returned.

    Returns:

    A tensor containing the entropy computed for all times for all features

    """

    missing = tensor_container.missing

    tensor_container.tensor = np.where(tensor_container.tensor==missing,0.0,tensor_container.tensor)

    tensor_container.tensor += offset

    entropy = np.zeros_like(tensor_container.tensor)

    for itime in range(tensor_container.tensor.shape[0]):
        if itime < window - 1:
            istart = 0
        else:
            istart = itime - window + 1

        sum_over_window = np.sum(tensor_container.tensor[istart:itime+1], axis=0)

        entropy[itime, :, :] = -np.sum(tensor_container.tensor[istart:itime+1]/sum_over_window *
                                       np.log2(tensor_container.tensor[istart:itime+1]/sum_over_window), axis=0)

    entropy += offset*np.log2(offset/window)

    tensor_container.tensor = entropy

    return tensor_container


@dne_wrapper
def temporal_tree_lag(tensor_container, index, thetacrit, weight_functions, sigma):
    """
    get_tree_lag

    Driver function for computing temporal-tree-lagged features

    Arguments are:

    tensor: tensor containing one or more features to be lagged
    thetacrit: parameter controlling how aggressively nodes in the past are aggregated
    weight_functions: list containing one or more of the following strings:

    - 'uniform': weights for all nodes are unity. Unlikely to be meaningful but provided for completeness
    - 'oneovert': weights for nodes are 1/(tnow-tnode)
    - 'expon': weights for nodes are exp(-(tnow-tnode)/sigma)
    - 'ramp': weights for nodes are 1-(tnow-tnode)/sigma for (tnow-tnode)<sigma, 0 otherwise
    - 'sigmoid': weights are 1./(1+np.exp(-lag)) where lag=(mid-tnow+5*sigma5)/sigma5 and sigma5=sigma/5

    sigma: parameter in time units used by df controlling width of expon, ramp and sigmoid functions

    """

    class TemporalNode():
        """
        TemporalNode() class
        Defines nodes for insertion in TemporalTree() class

         Nodes have:
        - nodeid: a unique id
        - level: integer indicating which level of the tree the node belongs to, 0 is leaf level
        - parent: parent node
        - start, end: node boundaries
        - sibling: id of node's sibling
        - predecessor: id of immediately preceding node at same tree level
        - ispast: flag indicating whether node is past or future compared to its sibling
        - isleaf: a flag indicating whether a leaf node
        - nleaf: the number of leaf nodes contained within the node's boundaries
        - children: list of node's children
        - features: dict of features loaded into the node by the .stock() method

        """

        def __init__(self, nodeid, level, start, end, parent, sibling, predecessor, ispast, isleaf):
            self.nodeid = nodeid
            self.level = level
            self.start = start
            self.end = end
            self.parent = parent
            self.sibling = sibling
            self.predecessor = predecessor
            self.ispast = ispast
            self.isleaf = isleaf
            self.children = []
            self.nleaf = 0
            self.features = {}

        def __repr__(self):
            return f""" nodeid: {self.nodeid}\n level: {self.level}\n parent: {self.parent}\n start: {self.start}
            end: {self.end}\n bottom: {self.bottom}\n isleaf: {self.isleaf}\n children: {self.children}
            predecessor: {self.predecessor}\n sibling: {self.sibling}\n"""

    class TemporalTree():
        """
        TemporalTree() class
        """

        def __init__(self):
            self.gridsize = 0
            self.nodes = []
            self.pgids = None
            self.maxlevel = -1
            self.stocked = False
            self.stocked_until = -1
            self.weight_fn = None
            self.timesequence = []
            self.npad = 0
            self.times = []
            self.time_to_index = {}
            self.index_to_time = {}
            self.pgid_to_index = {}
            self.index_to_pgid = {}
            self.features = {}
            self.weight_functions = {}

        def build_tree(self, index):

            """
            build_tree

            This function builds the tree from the leaf nodes downwards.

            The total number of leaf nodes must be an integer power of two, so the list of
            times is padded with dummy nodes placed before the earliest time the tree is
            built on.

            Once leaf nodes are populated, work down through the levels. Assign parents,
            siblings and predecessors

            """

            self.times, self.time_to_index, self.index_to_time = utilities.map_times(index)

            tstart = self.times[0]
            tend = self.times[-1]

            nseq_initial = tend - tstart
            log2_nseq_initial = np.log2(nseq_initial)
            nseq = int(2 ** (1 + int(log2_nseq_initial)))
            npad = nseq - nseq_initial

            self.timesequence = [t for t in range(tstart - npad, tend + 1)]
            self.npad = npad

            nodestodo = []
            nnodes = 0
            level = 0
            maxlevel = 0
            parent = -1
            sibling = -1
            predecessor = -1
            ispast = False
            isleaf = False
            node = TemporalNode(nnodes, level, tstart - npad, tend, parent, sibling, predecessor,
                                ispast, isleaf)
            self.nodes.append(node)
            nodestodo.append(node)

            while len(nodestodo) > 0:
                splitnode = nodestodo.pop()

                if (splitnode.end - splitnode.start) > 1:
                    nnodes += 1
                    if splitnode.start >= 0:
                        mid = int((splitnode.start + splitnode.end + 1) / 2)
                    else:
                        mid = int((splitnode.start + splitnode.end) / 2)

                    level = splitnode.level + 1
                    maxlevel = max(maxlevel, level)
                    parent = splitnode.nodeid
                    sibling = None
                    predecessor = None
                    ispast = True
                    if (mid - splitnode.start) > 1:
                        isleaf = False
                    else:
                        isleaf = True
                    pastnode = TemporalNode(nnodes, level, splitnode.start, mid, parent, sibling,
                                            predecessor, ispast, isleaf)

                    self.nodes.append(pastnode)

                    nodestodo.append(pastnode)

                    nnodes += 1
                    ispast = False
                    futurenode = TemporalNode(nnodes, level, mid, splitnode.end, parent, sibling,
                                              predecessor, ispast, isleaf)

                    self.nodes.append(futurenode)

                    nodestodo.append(futurenode)

                    pastnode.sibling = futurenode.nodeid

                    futurenode.sibling = pastnode.nodeid

                    futurenode.predecessor = pastnode.nodeid

                    splitnode.children = [pastnode.nodeid, futurenode.nodeid]

            self.maxlevel = maxlevel

            for node in self.nodes:
                while node.predecessor is None:
                    if node.start <= (self.timesequence[0] + self.npad):
                        node.predecessor = -1
                    else:
                        level = node.level
                        climb = True
                        climbnode = self.nodes[node.parent]
                        while climb:
                            if climbnode.ispast:
                                climbnode = self.nodes[climbnode.parent]
                            else:
                                descendnode = self.nodes[climbnode.sibling]
                                climb = False
                        descendlevel = descendnode.level

                        while descendnode.level != level:
                            descendnode = self.nodes[descendnode.children[1]]
                        node.predecessor = descendnode.nodeid

        def stock_initial(self, tensor):

            """
            stock_initial

            This function loads features into the tree.

            Data from the input df is first cast to a 3D tensor with dims time x pgid x
            features.

            The tensor is used to populate the leaf nodes, and values then just propagate
            down through the tree via the parents.

            At present, values of parents are the *sums* of those of their children.

            """

            if self.stocked:
                print('Tree has already been stocked - aborting')
                raise (Exception)

            self.stocked = True

            for node in self.nodes:
                for feature in range(tensor.shape[-1]):
                    node.features[feature] = np.zeros(tensor.shape[1])

            for time in self.times:
                itime = self.time_to_index[time]
                for node in self.nodes:
                    if node.isleaf and (node.start == time):
                        for ifeature in range(tensor.shape[-1]):
                            vals = tensor[itime, :, ifeature]

                            vals[vals == np.inf] = 0.0
                            vals[vals == -np.inf] = 0.0
                            vals[vals == np.nan] = 0.0

                            node.features[ifeature] += vals
                            node.nleaf = 1
                            parentid = node.parent
                            while parentid != -1:
                                parent = self.nodes[parentid]
                                parent.features[ifeature] += vals
                                parent.nleaf += 1
                                parentid = parent.parent

            self.stocked_until = self.times[-1]

        def walk(self, tnow, thetacrit):

            """
            walk

            This function generates the list of nodes any given node will import data from,
            when one lags variables using the tree, as well as weights based on the times
            between nodes' midpoints.

            The arguments are

            - thetacrit: angle used to decide if a candidate node should be added to a target
              node's interaction list, based on the size of the candidate node and
              the time gap between the candidate node and the target node

            """

            if (tnow < self.nodes[0].start) or (tnow > self.nodes[0].end):
                print('tnow not in range of times covered by this tree - aborting')
                raise Exception
            if tnow > self.stocked_until:
                print('tree has not been stocked as far as tnow - aborting')
                raise Exception

            list_of_nodes = []

            for node in self.nodes:
                if node.isleaf and (node.start == tnow):
                    list_of_nodes.append(node.nodeid)
                    if node.predecessor == -1:
                        return list_of_nodes
                    notdone = True
                    while notdone:
                        if node.ispast:
                            if node.predecessor == -1:
                                notdone = False
                            else:
                                pred = self.nodes[node.predecessor]
                                node = self.nodes[pred.parent]
                                self.split_node(node, list_of_nodes, tnow, thetacrit)

                        else:
                            node = self.nodes[node.sibling]
                            self.split_node(node, list_of_nodes, tnow, thetacrit)
                            node = self.nodes[node.parent]
                            if node.predecessor == -1:
                                notdone = False
                            else:
                                if node.sibling != node.predecessor:
                                    node = self.nodes[node.predecessor]
                                    self.split_node(node, list_of_nodes, tnow, thetacrit)

            return list_of_nodes

        def split_node(self, node, list_of_nodes, tnow, thetacrit):

            """
            split_node

            Function which decides whether or not to split a given node into its children,
            based on the critical angle and the current time
            """

            nodestocheck = []
            nodestocheck.append(node)
            while len(nodestocheck) > 0:
                node = nodestocheck.pop(0)
                mid = (node.start + node.end) / 2.
                width = (node.end - node.start)
                age = tnow - mid
                theta = width / age

                if theta < thetacrit:
                    list_of_nodes.append(node.nodeid)
                else:
                    if len(node.children) > 0:
                        nodestocheck.append(self.nodes[node.children[0]])
                        nodestocheck.append(self.nodes[node.children[1]])
                    else:
                        list_of_nodes.append(node.nodeid)

#        def make_weight_functions(self):
#            self.weight_functions['uniform'] = self._uniform
#            self.weight_functions['oneovert'] = self._oneovert
#            self.weight_functions['sigmoid'] = self._sigmoid
#            self.weight_functions['expon'] = self._expon
#            self.weight_functions['ramp'] = self._ramp

        def _uniform(self, node_list, tnow, sigma=1):
            weights = np.ones(len(node_list))
            return weights

        def _oneovert(self, node_list, tnow, sigma=1):
            weights = np.zeros(len(node_list))
            for i in range(len(node_list)):
                mid = (self.nodes[node_list[i]].start + self.nodes[node_list[i]].end) / 2.
                lag = tnow - mid + 1.5
                weights[i] = 1. / lag
            return weights

        def _sigmoid(self, node_list, tnow, sigma=1):
            weights = np.zeros(len(node_list))
            offset = 5.
            sigma /= offset
            for i in range(len(node_list)):
                mid = self.nodes[node_list[i]].start
                lag = (mid - tnow + offset * sigma) / sigma
                weights[i] = 1. / (1 + np.exp(-lag))

            return weights

        def _expon(self, node_list, tnow, sigma=1):
            weights = np.zeros(len(node_list))
            for i in range(len(node_list)):
                mid = (self.nodes[node_list[i]].start + self.nodes[node_list[i]].end) / 2.
                lag = tnow - mid
                lag1 = tnow - self.nodes[node_list[i]].start

                lag2 = tnow - self.nodes[node_list[i]].end

                w = np.exp(-lag / sigma)
                w1 = np.exp(-lag1 / sigma)
                w2 = np.exp(-lag2 / sigma)
                weights[i] = (8 * w1 + 6 * w - w2) / 13.

            return weights

        def _ramp(self, node_list, tnow, sigma=1):
            weights = np.zeros(len(node_list))
            for i in range(len(node_list)):
                mid = (self.nodes[node_list[i]].start + self.nodes[node_list[i]].end) / 2.
                lag = tnow - mid + 0.5
                weights[i] = 1. - (lag / sigma)

                weights[weights < 0.0] = 0.0
            return weights

        def get_weight_functions(self,weight_function_option):
                self.weight_functions = {}

                match (weight_function_option[0]):
                    case ('uniform'):

                         self.weight_functions['uniform'] = self._uniform

                    case('oneovert'):

                         self.weight_functions['oneovert'] = self._oneovert

                    case ('sigmoid'):

                         self.weight_functions['sigmoid'] = self._sigmoid

                    case ('expon'):

                        self.weight_functions['expon'] = self._expon

                    case ('ramp'):

                        self.weight_functions['ramp'] = self._ramp

        def tree_lag(self, tensor, thetacrit, sigma):

            """
            tree_lag

            This function computes features temporally lagged using the tree for every pgid.

            The interaction lists and weights, and the features loaded into each node are
            combined in weighted sums.

            The lagged features are then packed into a dataframe.

            """

            for time in self.times:
                itime = self.time_to_index[time]
                list_of_nodes = self.walk(time, thetacrit)
                for ifunction, weight_function in enumerate(self.weight_functions.keys()):
                    weights = self.weight_functions[weight_function](list_of_nodes, time, sigma)
                    for ifeature in range(tensor.shape[-1]):
                        weighted_feature = np.zeros_like(self.nodes[0].features[ifeature])
                        for nodeid, weight in zip(list_of_nodes, weights):
                            weighted_feature += weight * self.nodes[nodeid].features[ifeature]

                        tensor[itime, :, ifeature] = weighted_feature

            return tensor

    if type(weight_functions) != 'list':
        weight_functions = [weight_functions, ]

    tree = TemporalTree()

    tree.build_tree(index)

    tree.stock_initial(tensor_container.tensor)

    tree.get_weight_functions(weight_functions)

    tensor_container.tensor = tree.tree_lag(tensor_container.tensor, thetacrit, sigma)

    return tensor_container
