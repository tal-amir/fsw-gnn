# Injective Sliced-Wasserstein Embedding for Multisets and Weighted Point-Clouds

# Authors: Tal Amir, Nadav Dym
# Technion Institute of Technology
# Haifa, Israel

version = '1.28t2'
version_date = '2024-07-22'

# Changelog:
# 1.28t2  Testing cumsum_segments using jit
# 1.28t1  Testing sparse_cumsum_alt1
# 1.27    Made some slow assersions run only when sw_embedding_debug_mode or sw_embedding_basic_safety_checks are True
# 1.26    Removed the sp_old class. Got rid of more coalesce()
# 1.25c   Got rid of some more coalesce()
# 1.25b   Got rid of some more coalesce()
# 1.25    Removed most of coalesce() calls that follow calls to torch.sparse_coo_tensor()
# 1.24    Removed attributes 'device' and 'dtype' from SW_embedding due to lack of safety. Use get_device(), get_dtype() instead.
# 1.23    Added project_W()
# 1.22    Added support for biases
#         learnable_freqs=True now initializes frequencies to zero
# 1.21    Added reset_parameters()
#         Added type hinting and enforcement


import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.autograd.function import once_differentiable

import warnings
import numbers
import type_enforced # A runtime error in this line implies that that some function below is given an input arguemnt of the wrong type

import time

# Turn this on to run some verifications and sanity checks during runtime.
# If an error is encountered, a runtime error is raised
sw_embedding_debug_mode = False

# Conduct basic safety checks, mainly on the user input.
# Recommended to leave True, unless running time is of utmost importance, and the input is known to be consistent.
# Examples of safety checks:
# - Ensure that X and W do not contain infs and nans, and W only contains nonnegative values
# - Ensure that all multisets/distributions represented by W have a positive total mass (i.e. not zero total mass or empty multisets)
sw_embedding_basic_safety_checks = True


''' Maps multisets in R^d to vectors in R^m using the Sliced-Wasserstrin Embedding.
    Also supports weighted point-clouds in R^d, which are regarded as discrete distributions over R^d.
    
    The Euclidean distance between two embeddings approximates the Sliced-Wasserstein distance
    distance the input distributions:
                ||embed(X1,W1)-embed(X2,W2)||_2  =<approx>=  sqrt(m) * SW((X1,W1),(X2,W2))

    To guarantee that the embedding is injective with at most n input points, use
    m >= 2nd+1 for multisets and m >= 2n(d+1)+1 for distributions.
    
    The input point tensor X should be of size (<batch_dims>, n, d), where <batch_dims> can be
    any list of dimensions. The accompanying weights W should be of size (<batch_dims>, n).
    The output is of size (<batch_dims>, m).

    If W is not provided, all weights are assumed to be uniform 1/n, with n being the number
    of points in X.
            
    The weights should be non-negative. They do not have to be normalized, as they are normalized
    internally, but they should have a positive sum. 

    Graph mode
    ----------
    If graph_mode=True, W is treated as a conjugacy matrix of a graph with vertex features given in X.
    If X is of size (<batch_dims>, n, d), then W should be of size (<batch_dims>, nRecipients, n),
    with W[<batch_indices>, i, j] containing the weight of the edge from vertex j to vertex i. 
    The output is then of size (<batch_dims>, nRecipients, m), with each vector output[<batch indices>, i, :]
    holding the embedding of all feature vectors of neighbors of vertex i, with the corresponding weights being
    the weights of the edges leading from them to i.

    Note that W does not have to be square; hence the number of message recipients needs not be equal to the number
    of senders.

    Cartesian mode
    --------------
    If m=None and instead nSlices and nFreqs are provided, the embedding is computed with a Cartesian product of the
    slices and frequencies. The output shape is then (<batch_dims>, nSlices, nFreqs), or in graph mode
    (<batch_dims>, nRecipients, nSlices, nFreqs).
    
    If collapse_freqs=True, then the frequency axis is collaped to the slice axis, resulting in output size
    (<batch_dims>, nSlices x nFreqs), or in graph mode (<batch_dims>, nRecipients, nSlices x nFreqs).

    Sparse W
    --------
    The input W can be sparse. In some use cases this could lead to a considerable reduction in running time and memory
    complexity. The most common use scenario is in graph mode, when W represents the adjacency matrix of a graph with
    a large number of vertices and a relatively low number of edges.
'''

class SW_embedding(nn.Module):
    
    @type_enforced.Enforcer(enabled=True)
    def __init__(self, 
                 d : int, m : int | None = None,
                 nSlices : int | None = None, nFreqs : int | None = None, collapse_freqs : bool = False,
                 learnable_slices : bool = False, learnable_freqs : bool = False,
                 freqs_init : float | int | str | tuple[float,float] = 'random',
                 minimize_slice_coherence : bool = False,
                 enable_bias : bool = True,
                 device : torch.device | int | str | None = None, dtype : torch.dtype = torch.float32, 
                 report : bool = False,
                 report_on_coherence_minimization : bool = False):

        super().__init__()

        # Process sizes
        self.d = d

        input_space_name = 'R^%d' % (self.d)

        if (m is not None) and (nSlices is None) and (nFreqs is None):
            self.cartesian_mode = False            
            self.m = m
            self.nSlices = m
            self.nFreqs = m
            output_space_name = 'R^%d' % (self.m)
        elif (m is None) and (nSlices is not None) and (nFreqs is not None):
            self.cartesian_mode = True
            self.nSlices = nSlices
            self.nFreqs = nFreqs
            self.m = nSlices * nFreqs
            self.collapse_freqs = collapse_freqs
            output_space_name = ('R^%d' % (self.m)) if self.collapse_freqs else ('R^(%d\u00d7%d)' % (self.nSlices, self.nFreqs))
        else:
            assert False, "Expected exactly one of (m != None) or (nSlices and nFreqs != None)"

        m = self.m
        nSlices = self.nSlices
        nFreqs = self.nFreqs

        self.minimize_slice_coherence = minimize_slice_coherence

        self.learnable_slices = learnable_slices
        self.learnable_freqs = learnable_freqs

        # Note: freqs_init is checked for correctness downstream at generate_embedding_parameters()
        self.freqs_init = freqs_init

        self.enable_bias = enable_bias

        # device_new and dtype_new are only defined here on __init__ and passed on to reset_parameters(), which then deletes them
        self.device_new = ifnone(device, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        assert dtype.is_floating_point and (not dtype.is_complex), 'dtype must be real floating-point; instead got dtype=%s' % (dtype)
        self.dtype_new = dtype

        self.report = report
        self.report_on_coherence_minimization = report_on_coherence_minimization

        qprintln(report)
        qprintln(report, 'Sliced-Wasserstein Embedding')
        qprintln(report, 'version %s, %s' % (version, version_date))

        qprintln(report)
        qprintln(report, 'Tal Amir, Nadav Dym')
        qprintln(report, 'Technion Institute of Technology, Haifa, Israel')

        qprintln(report)
        qprintln(report, 'Based on our paper titled "Injective Sliced Wasserstrin Embedding for Multisets and Weighted Point Clouds", 2024')

        qprintln(report)
        qprintln(report, 'Constructing embedding for sets in %s into %s  ' % (input_space_name, output_space_name))

        if self.cartesian_mode and self.collapse_freqs:
            slice_freq_str = 'Using %d slices \u00d7 %d frequencies, collapsed to one %d dimensional axis; ' % (nSlices, nFreqs, nSlices*nFreqs)
        elif self.cartesian_mode:
            slice_freq_str = 'Using %d slices \u00d7 %s frequencies; ' % (nSlices, nFreqs)
        else:
            slice_freq_str = 'Using %d (slice, frequency) pairs; ' % (nSlices)

        qprint(report, slice_freq_str)

        if self.learnable_slices and self.learnable_freqs:
            if self.enable_bias:
                learnable_str = 'learnable slices, frequences and biases'
            else:
                learnable_str = 'learnable slices and frequences, no bias'
        elif self.learnable_slices:
            if self.enable_bias:
                learnable_str = 'learnable slices and biases, fixed frequencies'
            else:
                learnable_str = 'learnable slices, fixed frequences, no bias'
        elif self.learnable_freqs:
            if self.enable_bias:
                learnable_str = 'fixed slices, learnable frequencies, fixed biases (initialized to zero)'
            else:
                learnable_str = 'fixed slices, learnable frequencies, no biases'
        else:
            if self.enable_bias:
                learnable_str = 'fixed slices and frequencies, fixed biases (initialized to zero)'
            else:
                learnable_str = 'fixed slices and frequencies, no bias'

        qprintln(report, learnable_str)

        qprintln(report, 'device: %s    dtype: %s' % (self.device_new, self.dtype_new))

        self.reset_parameters()


    # Resets the model parameters (projection vectors and frequencies) and updates the model settings.
    @type_enforced.Enforcer(enabled=True)
    def reset_parameters(self,
                         freqs_init : float | int | str | None | tuple[float,float] = None,
                         minimize_slice_coherence : bool | None = None,
                         report : bool | None = None,
                         report_on_coherence_minimization : bool | None = None):

        # Apply user updates for these parameters
        self.freqs_init = ifnone(freqs_init, self.freqs_init)
        self.minimize_slice_coherence = ifnone(minimize_slice_coherence, self.minimize_slice_coherence)
        self.report = ifnone(report, self.report)
        self.report_on_coherence_minimization = ifnone(report_on_coherence_minimization, self.report_on_coherence_minimization)

        # To make sure we don't use these values inside the function; if any of then is None, we must use its self. counterpart.
        del minimize_slice_coherence, freqs_init, report, report_on_coherence_minimization

        qprintln(self.report)

        if hasattr(self, 'device_new'):
            qprintln(self.report, 'Generating embedding parameters:')
        else:
            qprintln(self.report, 'Resetting embedding parameters:')


        # If we're running for the first time, get the device and dtype that were set in the __init__ method;
        # otherwise use the current device and dtype.
        if hasattr(self, 'device_new'):
            device = self.device_new
            delattr(self, 'device_new')
        else:
            device = self.get_device()

        if hasattr(self, 'dtype_new'):
            dtype = self.dtype_new
            delattr(self, 'dtype_new')
        else:
            dtype = self.get_dtype()

        # Generate projection vectors and frequencies
        # We always generate (and optimize) them in float64 and then convert to the desired dtype.
        projVecs, freqs, bias = SW_embedding.generate_embedding_parameters(d=self.d, 
                                                                           nSlices=self.nSlices, nFreqs=self.nFreqs,
                                                                           cartesian_mode=self.cartesian_mode,
                                                                           freqs_init=self.freqs_init,
                                                                           minimize_slice_coherence=self.minimize_slice_coherence,
                                                                           device=device,
                                                                           report = self.report,
                                                                           report_on_coherence_minimization = self.report_on_coherence_minimization)

        projVecs = projVecs.to(dtype=dtype, device=device)
        freqs = freqs.to(dtype=dtype, device=device)

        self.projVecs = nn.Parameter( projVecs, requires_grad=self.learnable_slices )
        self.freqs = nn.Parameter( freqs, requires_grad=self.learnable_freqs )

        if self.enable_bias:
            bias = bias.to(dtype=dtype, device=device)

            if self.cartesian_mode and self.collapse_freqs:
                bias = bias.reshape((self.nSlices*self.nFreqs))

            self.bias = nn.Parameter( bias, requires_grad=self.learnable_slices )

        # This also initializes the .device and .dtype fields
        self.to(device=self.get_device(), dtype=self.get_dtype())

        return self


    def to(self, *args, **kwargs):
        if 'dtype' in kwargs:
            arg = kwargs['dtype']

            assert isinstance(arg, torch.dtype), 'invalid input type %s at argument ''dtype''' % (type(arg))
            assert arg.is_floating_point and not arg.is_complex, 'dtype must be real floating-point; instead got dtype=%s' % (arg)

        for arg in args:
            if isinstance(arg, torch.dtype):
                assert arg.is_floating_point and not arg.is_complex, 'dtype must be real floating-point; instead got dtype=%s' % (arg)

        super().to(*args, **kwargs)       

        # Note: We never rely on self.device internally. We just update it for the user.
        # TODO: These are unsafe. If a parent module T has an SW_embedding as a submodule, e.g. T.swembed,
        #       then calling T.to(dtype=something) does not recursively call T.swembed.to, but does recursively
        #       change the dtype of the tensors in T.swmembed (e.g. projVecs, freqs). In this case, T.swembed.dtype
        #       does not get updated to the true dtype (although T.swembed.get_dtype() does return the true dtype).
        # Therefore I'm disabling them until I can ensure that they're always in a consistent state.
        # self.device = self.get_device()
        # self.dtype = self.get_dtype()
        
        return self


    def get_device(self):
        return self.projVecs.device


    def get_dtype(self):
        return self.projVecs.dtype
    

    def generate_embedding_parameters(d, nSlices, nFreqs, cartesian_mode, freqs_init, minimize_slice_coherence, device, report, report_on_coherence_minimization):
        dtype_init = torch.float64

        # Axis number for the ambient space R^d
        ambspace_axis = 1

        ### A. Generate projection vectors

        projVecs = torch.randn(size=(nSlices, d), dtype=dtype_init, device=device)
        projVecs = nn.functional.normalize(projVecs, p=2.0, dim=ambspace_axis, eps=0, out=None)

        if minimize_slice_coherence:
            if nSlices > d:
                projVecs = minimize_mutual_coherence(projVecs, report=report_on_coherence_minimization)
                qprintln(report, '- Generated %d projection vectors in R^%d with minimized mutual coherence' % (nSlices, d))
            else:
                projVecs, _ = torch.linalg.qr(projVecs.transpose(0,1), mode='reduced')
                projVecs = projVecs.transpose(0,1)
                qprintln(report, '- Generated %d perpendicular projection vectors in R^%d using QR decomposition' % (nSlices, d))
        else:
            qprintln(report, '- Generated %d random projection vectors' % (nSlices))

        # Detect nans, infs and zero vectors in projVecs
        assert not torch.isinf(projVecs).any(), "Found infs in projVecs"
        assert not torch.isnan(projVecs).any(), "Found nans in projVecs"
        assert not (projVecs == 0).all(dim=1).any(), 'Found zero vectors in projVecs'


        ### B. Generate frequencies
        freqs_shape = (nFreqs,) # Note: Changing this to (self.nFreqs, 1) yields incorrect results in self.forward()

        if nFreqs == 0:
            freqs = torch.zeros(size=freqs_shape, dtype=dtype_init, device=device)
            qprintln(report, '- Initialized 0 frequencies')

        elif isinstance(freqs_init, numbers.Real):
            assert not np.isinf(freqs_init), 'freqs_init cannot be infinite'
            assert not np.isnan(freqs_init), 'freqs_init cannot be NaN'
            freqs = freqs_init * torch.ones(size=freqs_shape, dtype=dtype_init, device=device)
            qprintln(report, '- Initialized %d frequencies to %g' % (nFreqs, freqs_init))

        elif isinstance(freqs_init, tuple):            
            # Here freqs_init should have been type-enforced to be a tuple of two real numbers.
            # However, it does not prevent the tuple from containing more numbers.
            assert(len(freqs_init)==2), 'When freqs_init is a tuple, it must be of length 2'

            a = freqs_init[0]
            b = freqs_init[1]
            assert not np.isinf(a) and not np.isinf(b), 'Received infinite value in freqs_init tuple'
            assert not np.isnan(a) and not np.isnan(b), 'Received NaN value in freqs_init tuple'
            assert a <= b, 'When freqs_init is a tuple, it is required to satisfy freqs_init[0] <= freqs_init[1]'

            if nFreqs == 1:
                freqs = a + (b-a)/2 * torch.ones(size=freqs_shape, dtype=dtype_init, device=device)
            else:
                freqs = a + (b-a) * ( torch.arange(nFreqs, dtype=dtype_init, device=device) / (nFreqs-1) )

            qprintln(report, '- Initialized %d equispaced frequencies in the interval [%d, %d]' % (nFreqs,a,b))

        elif freqs_init == 'random':
            freqs = torch.rand(size=freqs_shape, dtype=dtype_init, device=device)
            freqs, junk = torch.sort(freqs, dim=0)
            assert (freqs != 1).all(), "Unexpected behavior of torch.rand(): Returned a value of 1, whereas values are supposed to be in [0,1)"
            assert (freqs < 1).all(), "Unexpected behavior of torch.rand(): Returned a value > 1, whereas values are supposed to be in [0,1)"
            freqs = freqs / (1-freqs)

            qprintln(report, '- Initialized %d random frequencies i.i.d. with density f(x) = 1/(1+x)^2, x\u22650' % (nFreqs))

        elif freqs_init == 'spread':
            freqs = ( 0.5 + torch.arange(nFreqs, dtype=dtype_init, device=device).reshape(freqs_shape) ) / nFreqs
            freqs = freqs / (1-freqs)
            qprintln(report, '- Initialized %d frequencies spread evenly in [%g, %g] according to probability density' % (nFreqs, freqs[0], freqs[-1]))

        else:
            raise RuntimeError('Invalid value for argument freqs_init; expected number, tuple (a,b) of numbers denoting an interval, \'random\' or \'spread\'')
        
        # Detect nan and inf entries in freqs
        if nFreqs > 0:
            assert not torch.isinf(freqs).any(), "Found infs in freqs"
            assert not torch.isnan(freqs).any(), "Found nans in freqs"

        # C. Generate bias vector. Always initialized to zero.
        bias_shape = (nSlices, nFreqs) if cartesian_mode else (nSlices,)
        bias = torch.zeros(size=bias_shape, dtype=dtype_init, device=device)

        qprintln(report)

        return projVecs, freqs, bias


    # Spreads the frequencies on an interval centered at 'center' with the given radius, in an equispaced manner.
    # This might be useful when using the embedding for graph message passing with learnable_projs=True, as the magnitude of the
    # projection vectors already determines the effective frequency, and having a very high max-frequency-to-low-frequency ratio
    # may impede the optimization due to ill conditioning.
    @type_enforced.Enforcer(enabled=True)
    def spread_freqs_at_interval(self, center : float | int, radius : float | int):
        assert radius >= 0

        if (self.nFreqs == 1) or (radius == 0):
            freqs_new = center * torch.ones_like(self.freqs)
        else:
            spread = 2 * (0.5 + torch.arange(self.nFreqs, dtype=self.get_dtype(), device=self.get_device()).reshape(self.freqs.shape) ) / self.nFreqs - 1
            spread = spread * 1/(1 - 1/self.nFreqs)
            freqs_new = center + radius * spread

        state_dict = self.state_dict()
        state_dict['freqs'] = freqs_new
        self.load_state_dict(state_dict)

        return self


    @type_enforced.Enforcer(enabled=True)
    def forward(self, X, W=None, graph_mode : bool = False, serialize_num_slices : int | None = None):
        # Simple use:
        # X sized (n,d) represents a multiset of n points in R^d.
        # If W sized (n,) is provided, then (X,W) represents a distribution, with each point X[i,:]
        # assigned the weight W[i]. 
        # The weights are normalized internally so they do not have to sum up to 1, but they must be nonnegative
        # and contain at least one nonzero.
        # If W is not provided, it is assumed to be uniform weights 1/n.
        # The output embedding is of size (m,).

        # Batches:
        # Batches of distributions as above can be provided via X of size (<batch_dims>, n, d) and W (<batch_dims>, n).
        # Here the output will be of size (<batch_dims>, m).

        # Graph mode: (Requires that W be given explicitly)
        # If graph_mode=True, then the points in X are shared between all batches along the last batch dimension.
        # That is, forward(X,W,graph_mode=True) produces (more efficiently) the same result as forward(X_expand, W),
        # where d = X.shape[-1]  and  X_expand = X.unsqueeze(dim=-3).expand( tuple(W.shape) + (d,) ).
        # A common usage for this feature is when W sized (n,n) represents an adjacency matrix of a graph, 
        # and X sized (n,d) represents vertex features. Then the output of embed(X,W,graph_mode=True) will be of size (n,m),
        # with each of its [i,:] rows representing the embedding of the features of all neighbours of vertex i, with weights
        # given by their edge weights.
        # The input in graph mode can be batched as follows: W of size (<batch_dims>, n) with X of size (batch_dims[:-1], d)
        # Note that batch_dims[-1] is not required to equal n, i.e. it is possible to process subblocks of adjacency matrices.

        # Output:
        # X_emb: The Sliced-Wasserstein embedding of the input distributions.
        #
        # By default the output size is (<batch_dims>, m), where each X_emb(j1,...,jk,:) contains the embedding of one distribution.
        #
        # If the parameters nSlices and nFreqs are set on initialization, then the output is of size
        # (<batch_dims>, nSlices, nFreqs) or (<batch_dims>, nSlices*nFreqs) - the former if 
        # the parameter 'collapse_freqs' is set to False (default), the latter if it is set to True.
        #
        # If serialize_num_slices = t (integer), then the computation is serialized to batches of size t.
        # This does not affect the result, but reduces the momery complexity by a factor of <number of slices> / t

        # Verify slices and frequencies at each forward pass if they are learnable
        if sw_embedding_basic_safety_checks and self.learnable_slices:
            assert not torch.isnan(self.projVecs).any(), 'Projection vectors contain NaNs'
            assert not torch.isinf(self.projVecs).any(), 'Projection vectors contain infs'
            # Note: We allow them to contain zero vectors when they are learnable, in case i.e. when sparsity is desired
            # assert not (self.projVecs == 0).all(dim=1).any(), 'Projection vectors contain a zero vector'

        if sw_embedding_basic_safety_checks and self.learnable_freqs:
            assert not torch.isnan(self.freqs).any(), 'Frequencies contain NaNs'
            assert not torch.isinf(self.freqs).any(), 'Frequencies contain infs'

        ### A. Verify input types and content

        assert torch.is_tensor(X), 'X must be a pytorch tensor. Instead got type %s' % (type(X))
        assert W is None or torch.is_tensor(W), 'W must be a pytorch tensor. Instead got type %s' % (type(W))
        assert X.dtype == self.get_dtype(), ( "X has the wrong dtype. Expected %s, got %s" % (self.get_dtype(), X.dtype) )
        assert X.device == self.get_device(), ( "X is on the wrong device. Expected %s, got %s" % (self.get_device(), X.device) )

        if sw_embedding_basic_safety_checks:
            assert not torch.isnan(X).any(), "The entries of X cannot contain NaNs"
            assert not torch.isinf(X).any(), "All entries of X must be finite"

        if W is not None:
            # Check if W is sparse. If so, ensure that W is of the correct layout.            
            # Note: Strangely enough, sparse tensors of layouts other than COO (e.g. CSR) may have is_sparse=False.
            #       This may lead us to mistakenly treat a, e.g. W that is sparse CSR as dense.
            #       Currently there is no is_dense() function in torch, so reading the layout string directly is the 2nd best.
            if W.is_sparse or W.layout != torch.strided:
                assert W.layout == torch.sparse_coo, ( "Sparse W has an unsupported sparsity layout '%s'. Only the COO layout (torch.sparse_coo) is currently supported." % (W.layout) )

                assert W.is_coalesced(), 'Sparse W must be coalesced'

                if sw_embedding_basic_safety_checks:
                    W_vals = W.values()
            else:
                if sw_embedding_basic_safety_checks:
                    W_vals = W

            assert W.dtype == self.get_dtype(), ( "W has the wrong dtype. Expected %s, got %s" % (self.get_dtype(), W.dtype) )
            assert W.device == self.get_device(), ( "W is on the wrong device. Expected %s, got %s" % (self.get_device(), W.device) )

            if sw_embedding_basic_safety_checks:
                assert not torch.isnan(W_vals).any(), "W cannot contain NaNs"
                assert not torch.isinf(W_vals).any(), "All entries of W must be finite"
                assert (W_vals >= 0).all(), "All entries of W must be nonnegative"
                assert (W_vals > 0).any(dim=-1).all(), "W assigns an all-zero weight to one of its distributions"

                if W.requires_grad and (W_vals == 0).any():
                    warnings.warn('Gradients of W may be incorrect at indices where W=0. Use W = SW_embedding.project_W(W, eps) with eps > 0')


        ### B. Verify input sizes
            
        assert len(X.shape) >= 2, "X must be a tensor of order at least 2"
        assert X.shape[-1] == self.d, "The last dimension of X must equal d=%d. Instead got %d" % (self.d, X.shape[-1])        

        if not graph_mode:
            # batch_dims contains everything that precedes (n,d) in X.shape
            batch_dims = tuple(X.shape[0:-2])
            n = X.shape[len(batch_dims)]

            if W is not None:
                if (len(W.shape) == len(X.shape)) and (W.shape[-1] == X.shape[-2]) and (W.shape[0:-2] == X.shape[0:-2]):
                    err_str = "Shape mismatch between X and W: If X.shape = (b1,b2,...,bk,n,d) then W.shape should be (b1,b2,...,bk,n) (Perhaps missing argument graph_mode=True?)"
                else:
                    err_str = "Shape mismatch between X and W: If X.shape = (b1,b2,...,bk,n,d) then W.shape should be (b1,b2,...,bk,n) (unless graph_mode=True)"
                
                assert (len(W.shape) == len(X.shape)-1) and (W.shape == X.shape[0:-1]), err_str

            else:
                # Initialize with uniform weights
                W = torch.ones(batch_dims + (n,), dtype=self.get_dtype(), device=self.get_device())

        elif graph_mode:
            assert W is not None, 'W must be explicitly provided when graph_mode=True'

            # batch_dims contains everything that precedes (nRecipients, n) in W.shape
            batch_dims = tuple(W.shape[0:-2])
            nRecipients = W.shape[-2]
            n = W.shape[-1]

            assert (len(W.shape) == len(X.shape)) and (W.shape[-1] == X.shape[-2]) and (W.shape[0:-2] == X.shape[0:-2]), "Shape mismatch between X and W: When graph_mode=True, if W.shape = (b1,b2,...,bk,nRecipients,n) then X.shape should be (b1,b2,...,bk,n,d)"


        ### C. Precalculate axis indices and output shape
             
        # These are the different axes we use to store data for processing. These definitions are repeated in forward_helper()
        # element_axis corresponds to the index of the multiset elements
        # ambspace_axis corresponds to the elements' coordinate index in the ambient space R^d
        # After projection, the ambient space coordinates are replaced by the projection/slice number; thus proj_axis=ambspace_axis
        # If we're in Cartesian mode, the frequencies have their own axis freq_axis, otherwise it is the same axis as proj_axis.
        recipient_axis = len(batch_dims) if graph_mode else None  # Message-recipient vertices
        element_axis  = recipient_axis+1 if graph_mode else len(batch_dims) # In graph mode this axis denotes the message-sender vertices
        ambspace_axis = element_axis + 1        
        proj_axis     = ambspace_axis
        freq_axis     = proj_axis +1 if self.cartesian_mode else proj_axis
        output_proj_axis = element_axis # In the output, the element axis is replaced by the projection axis

        output_shape_before_collapse =  batch_dims + (nRecipients,) if graph_mode else batch_dims
        output_shape_before_collapse += (self.nSlices, self.nFreqs) if self.cartesian_mode else (self.m, )

        ### D. Input is ok. Start working.
            
        # Normalize W
        if W.is_sparse:
            W_sum = ag.sum_sparseToDense.apply(W, -1)
            W = ag.div_sparse_dense.apply(W, W_sum)
            del W_sum
        else:
            W = W / torch.sum(W, dim=-1, keepdim=True)

        # For compatibility reasons, we support the case of zero-dimensional output tensor
        if self.m == 0:
            X_emb = torch.zeros(size=output_shape_before_collapse, dtype=self.get_dtype(), device=self.get_device())

        elif (serialize_num_slices is None) or (serialize_num_slices >= self.nSlices):
            X_emb = SW_embedding.forward_helper(X, W, self.projVecs, self.freqs, graph_mode, self.cartesian_mode, batch_dims)

        else:
            assert isinstance(serialize_num_slices, int) and (serialize_num_slices >= 1), 'serialize_num_slices must be None or a positive integer'

            nIter = (self.nSlices // serialize_num_slices) if (self.nSlices % serialize_num_slices == 0) else (1 + self.nSlices // serialize_num_slices)

            X_emb = torch.empty(size=output_shape_before_collapse, dtype=self.get_dtype(), device=self.get_device())

            for iIter in range(nIter):
                inds_curr = torch.arange(iIter*serialize_num_slices, min( self.nSlices, (iIter+1)*serialize_num_slices), dtype=torch.int64, device=self.get_device())
                projVecs_curr = self.projVecs[inds_curr,:]
                freqs_curr = self.freqs if self.cartesian_mode else self.freqs[inds_curr]
                
                out_curr = SW_embedding.forward_helper(X, W, projVecs_curr, freqs_curr, graph_mode, self.cartesian_mode, batch_dims)                
                assign_at(X_emb, out_curr, output_proj_axis, inds_curr)

        if self.cartesian_mode and self.collapse_freqs:
            X_emb = torch.flatten(X_emb, start_dim=element_axis, end_dim=element_axis+1)

        # Add bias
        if self.enable_bias:
            # Output bias shape before broadcasting
            bias_out_shape = (1,)*(X_emb.dim()-self.bias.dim()) + tuple(self.bias.shape)
            bias_reshape = torch.reshape(self.bias, bias_out_shape)            
            X_emb += bias_reshape

        return X_emb


    def forward_helper(X, W, projVecs, freqs, graph_mode, cartesian_mode, batch_dims):
        # This function computes the embedding of (X,W) for a subset of the projections and frequencies.
        # projVecs should be of size (num_projections x d), and freqs should be of size nFreqs (not nFreqs x 1).

        d = X.shape[-1]
        n = W.shape[-1]
        nProjs = projVecs.shape[0]
        nFreqs = len(freqs)
        sparse_mode = W.is_sparse

        assert len(freqs.shape) == 1, "This should not happen"
        assert (len(projVecs.shape) == 2) and (projVecs.shape[1] == d), "This should not happen"

        # Calculate the projections of X 
        Xp = torch.tensordot(X, projVecs, dims=((-1,),(1,)))

        del X

        # Sort the projected elements 
        # Note: We sort before the graph-mode expansion because it makes things simpler in the case when W is sparse
        if sparse_mode:
            Xps, Xpi = ag.sort.apply(Xp, -2, False)
        else:
            Xps, Xpi = torch.sort(Xp, dim=-2, descending=False)

        del Xp

        if graph_mode:
            Xps = Xps.unsqueeze(dim=-3)
            Xpi = Xpi.unsqueeze(dim=-3)

        # Axis numbers as in the implementation of forward()
        # Note: These numbers are true only from here
        recipient_axis = len(batch_dims) if graph_mode else None  # Message-recipient vertices
        element_axis  = recipient_axis+1 if graph_mode else len(batch_dims) # In graph mode this axis denotes the message-sender vertices
        ambspace_axis = element_axis + 1        
        proj_axis     = ambspace_axis
        freq_axis     = proj_axis +1 if cartesian_mode else proj_axis
        output_proj_axis = element_axis # In the output, the element axis is replaced by the projection axis

        assert len(freqs.shape) == 1
        for i in range(freq_axis):
            freqs = freqs.unsqueeze(0)
            
        if not sparse_mode:
            if graph_mode:
                Xps = Xps.expand(tuple(W.shape) + (nProjs,))
                Xpi = Xpi.expand(tuple(W.shape) + (nProjs,))

            # Sort the weights according to their corresponding projected elements
            W_big = W.unsqueeze(-1).expand_as(Xps)
            Wps = torch.gather(W_big, dim=element_axis, index=Xpi)

            if cartesian_mode:
                Wps = Wps.unsqueeze(dim=-1)
                Xps = Xps.unsqueeze(-1).expand(Xps.shape + (nFreqs,))

            # Once we have Wps we don't need W_big and Xpi
            del W_big, Xpi

            Wps_sum = torch.cumsum(Wps, dim=element_axis)

            # Here we assume sinc(x) = sin(pi*x)/(pi*x)
            sincs = 2 * Wps_sum * torch.sinc(2 * freqs * Wps_sum)
            sinc_diffs = diff_zeropad(sincs, dim=element_axis)
            del sincs

        elif sparse_mode:
            # We unsqueeze W to add a projection axis, in order to sort W according to each projection of X
            # Note: This repmat is unavoidable, because we sort the weights according to different permutations along proj_axis
            W_unsqueeze = ag.unsqueeze_sparse.apply(W,-1)
            del W

            W_big = ag.repmat_sparse.apply(W_unsqueeze, nProjs, proj_axis)
            del W_unsqueeze

            if graph_mode: 
                Wps = ag.permute_sparse.apply(W_big, element_axis, Xpi, recipient_axis)
            else:
                Wps = ag.permute_sparse.apply(W_big, element_axis, Xpi, None)

            # Once we have Wps we don't need W_big and Xpi
            del W_big, Xpi

            Wps_sum = ag.cumsum_sparse.apply(Wps, element_axis)

            if cartesian_mode:
                # TODO:
                # These repmats may be avoided if ag.sinc_cos_sparse could take freqs as a separate input, and broadcast all inputs accordingly.
                # But sinc_diffs is of the same size as Wps and Wps_sum, so we could reduce the memory usage at most by 2/3, and only in cartesian mode.
                # This may not worth the effort.

                Wps = ag.repmat_sparse.apply(ag.unsqueeze_sparse.apply(Wps,-1), nFreqs, freq_axis)
                Wps_sum = ag.repmat_sparse.apply(ag.unsqueeze_sparse.apply(Wps_sum,-1), nFreqs, freq_axis)
                Xps = Xps.unsqueeze(-1).expand(Xps.shape + (nFreqs,))

                

            # Here we use the sum-to-product identity sin(2a)-sin(2b) = 2*sin(a-b)*cos(a+b)            
            # This formula probably leads to a loss of one significant digit, but it is much easier in the sparse case than using diff().
            
            # Variant 1 is simpler to read, but requires more memory.
            # Variant 2 is more memory efficient.
            # Default: 2
            variant = 2

            if variant == 1:
                arg2 = np.pi * freqs * (2*Wps_sum - Wps)
                del Wps_sum
                assert_coalesced(arg2)
            elif variant == 2:                               
                #assert_debug( (arg2.indices()==Wps.indices()).all(), '' )

                # The command below is a more economic way of doing: arg2 = 2*Wps_sum - Wps
                arg2 = sp.sparse_coo_tensor_coalesced(indices=Wps.indices(), values=2*Wps_sum.values()-Wps.values(), size=Wps.shape)
                arg2 = ag.mul_sparse_dense.apply(arg2, np.pi*freqs)
            else:
                raise RuntimeError('This should not happen')

            arg1 = ag.mul_sparse_dense.apply(Wps, freqs)

            sinc_cos = ag.sinc_cos_sparse.apply(arg1, arg2)
            del arg1, arg2

            sinc_diffs = 2 * ag.mul_sparse.apply( Wps, sinc_cos )
            del Wps, sinc_cos
           
        # From here we only need sinc_diffs and Xps               

        if sparse_mode:
            products = ag.mul_sparse_dense.apply(sinc_diffs, Xps)
            product_sums = ag.sum_sparseToDense.apply(products, element_axis)
            
        else: # not sparse
            product_sums = torch.sum(sinc_diffs * Xps, dim=element_axis, keepdim=True)

        # We squeeze the element axis after having summed up along it
        product_sums = product_sums.squeeze(dim=element_axis)
        freqs = freqs.squeeze(dim=element_axis)

        out = (1+freqs) * product_sums

        del product_sums

        return out.to_dense()


    # Project W to the probability simplex. To be used in a Projected Gradient Descent scheme.
    @type_enforced.Enforcer(enabled=True)
    def project_W(W : torch.Tensor, eps : float = 1e-8):
        assert np.isfinite(eps), 'eps must be finite'
        assert not np.isnan(eps), 'eps cannot be NaN'
        assert eps >= 0, 'eps must be nonnegative'

        assert torch.is_floating_point(W), 'W must be a floating-point tensor'

        requires_grad_input = W.requires_grad

        # We don't want these actions to be recorded in the computation graph
        with torch.no_grad():
            if W.is_sparse or W.layout != torch.strided:
                assert W.layout == torch.sparse_coo, ( "Sparse W has an unsupported sparsity layout '%s'. Only the COO layout (torch.sparse_coo) is currently supported." % (W.layout) )

                assert W.is_coalesced(), 'Sparse W must be coalesced'
                inds = W.indices()
                vals = W.values()

                if sw_embedding_basic_safety_checks:
                    assert not torch.isinf(vals).any(), 'W cannot contain infinite values'
                    assert not torch.isnan(vals).any(), 'W cannot contain NaNs'

                vals = torch.clamp(vals, min=eps)

                W = sp.sparse_coo_tensor_coalesced(indices=inds, values=vals, size=W.shape)
                
                W_sum = ag.sum_sparseToDense.apply(W, -1)

                if sw_embedding_basic_safety_checks:
                    assert not (W_sum == 0).any(), "W assigns an all-zero weight to one of its distributions"

                W = ag.div_sparse_dense.apply(W, W_sum)
            
            else:
                if sw_embedding_basic_safety_checks:
                    assert not torch.isinf(W).any(), 'W cannot contain infinite values'
                    assert not torch.isnan(W).any(), 'W cannot contain NaNs'

                W = torch.clamp(W, min=eps)

                if sw_embedding_basic_safety_checks:
                    assert (W > 0).any(dim=-1).all(), "W assigns an all-zero weight to one of its distributions"

                W = W / torch.sum(W, dim=-1, keepdim=True)

            W.requires_grad = requires_grad_input
            return W



    def get_mutual_coherence(self):
        gram = self.projVecs @ self.projVecs.transpose(0,1)
        inds = range(self.m)
        gram[inds,inds] = 0

        mu = torch.max(torch.abs(gram))
        
        welch_bound = np.sqrt( (self.m - self.d) / self.d / (self.m-1) ) 
        d3_bound = -1.0 + 2.0 * (2.0*(1.0-1.0/self.m)**2 - 1)**2

        if self.d == 3:
            lower_bound = d3_bound
        else:
            lower_bound = welch_bound

        #return mu, lower_bound
        return mu




#############################################################################################################
##                                                  Tools                                                  ##
#############################################################################################################

def assert_coalesced(A):
    debug = sw_embedding_debug_mode
    if debug:
        assert A.is_coalesced(), 'tensor is not coalesced'

# Computes a finite difference with zero padding
def diff_zeropad(input, dim):
    pad_shape = replace_in_tuple(tuple(input.shape), index=dim, value=1)
    pad = torch.zeros(size=pad_shape, dtype=input.dtype, device=input.device)
    out = torch.diff(input, n=1, dim=dim, prepend=pad, append=None)
    return out


def replace_in_tuple(T, index, value):
    out = T[0:index] + (value,) + T[(index+1):len(T)]
    return out


@type_enforced.Enforcer(enabled=True)
def qprint(q : bool, s : str =''):
    assert type(q) == type(True)

    if q:
        print(s, end='')


@type_enforced.Enforcer(enabled=True)
def qprintln(q : bool, s : str =''):
    qprint(q, s+'\n')


# Performs something like target[:,:,...,:,inds,:,...,:] = source, where the argument 'inds' is given at dimension dim
def assign_at(target, source, dim, inds):
    scatter_inds_shape = replace_in_tuple((1,)*len(target.shape), dim, len(inds))               
    scatter_inds = inds.reshape(scatter_inds_shape).expand_as(source)
    target.scatter_(dim=dim, index=scatter_inds, src=source)
    

def ifnone(a,b):
    return a if (a is not None) else b



#############################################################################################################
##                                        Custom autograd functions                                        ##
#############################################################################################################

# These implementations were based on the example in
# https://pytorch.org/tutorials/beginner/examples_WithAutograd/two_layer_net_custom_function.html

# This code is based on Torch version 2.1.1, whose current support for gradients of sparse tensors sucks.
# Therefore I had to implement many of basic sparse-tensor operations used in this code myself.
# Should torch add autograd support for some of these operations in later versions, I will replace them adqeuately.

# Most actions here that take sparse tensor inputs require them to be coalesced, and actions
# that return sparse outputs return coalesced outputs.

# All custom autograd functions used here are defined under the 'ag' class.
class ag:
    # Permutes each 1-dimensional slice of A along dimension dim according to the given permutation in perms.
    # Perms can be broadcast to the size of A along dimension broadcast_perms_dim.
    class permute_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, dim, perms, broadcast_perms_dim):
            assert A.is_coalesced(), 'A must be coalesced'

            if ctx.needs_input_grad[0]:
                ctx.dim = dim if dim >= 0 else ( dim + A.dim() )
                ctx.broadcast_perms_dim = broadcast_perms_dim

                # Try to save space
                perms_max = A.shape[dim]-1
                #assert perms_max == perms.max() # Sanity check

                if perms_max <= torch.iinfo(torch.int16).max:
                    perms = perms.to(dtype=torch.int16)
                elif perms_max <= torch.iinfo(torch.int32).max:
                    perms = perms.to(dtype=torch.int32)

                ctx.save_for_backward(perms)

            out = sp.permute(A, dim=dim, perms=perms, broadcast_perms_dim=broadcast_perms_dim, backward_mode=False)

            return out


        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            if ctx.needs_input_grad[2]:
                assert False, 'This is strange. Autograd requires gradient of permute_sparse() wrt the permutations'

            if not ctx.needs_input_grad[0]:
                return None, None, None, None
            
            dim = ctx.dim
            broadcast_perms_dim = ctx.broadcast_perms_dim
            perms, = ctx.saved_tensors

            out = sp.permute(grad_output, dim=dim, perms=perms, broadcast_perms_dim=broadcast_perms_dim, backward_mode=True)

            return out, None, None, None


    # Equivalent to: torch.unsqueeze(dim)
    class unsqueeze_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, dim):
            assert A.is_sparse, 'A must be sparse'
            assert A.is_coalesced(), 'A must be coalesced'

            # Note that here it is dim+A.dim()+1 rather than the usual dim + A.dim(), as defined in the unsqueeze() command
            ctx.dim = dim if dim >= 0 else ( dim + A.dim() + 1 )
            ctx.input_shape = A.shape

            variant = 2

            if variant == 1:
                return A.unsqueeze(dim=dim)
            
            elif variant == 2:
                assert_coalesced(A)

                vals = A.values()
                
                inds = A.indices()
                inds = torch.cat( (inds[0:ctx.dim,:], torch.zeros((1,inds.shape[1]), device=inds.device, dtype=inds.dtype), inds[ctx.dim:,:]), dim=0 )

                output_shape = ctx.input_shape[0:ctx.dim] + (1,) + ctx.input_shape[ctx.dim:]

                return sp.sparse_coo_tensor_coalesced(indices=inds, values=vals, size=output_shape)



        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):        
            dim = ctx.dim          
            input_shape = ctx.input_shape

            input_dims_in_output = torch.arange(len(input_shape)+1, device=grad_output.device)
            input_dims_in_output = input_dims_in_output[input_dims_in_output != dim]

            inds = torch.index_select(grad_output.indices(), 0, input_dims_in_output)
            vals = grad_output.values()

            grad_input = sp.sparse_coo_tensor_coalesced(indices=inds, values=vals, size=input_shape)
            
            # Note: If there was a torch.squeeze() for sparse tensors, we could have done this in one line.

            return grad_input, None


    # Equivalent to: torch.sum(A, dim=dim, keepdim=True)
    class sum_sparseToDense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, dim):
            assert A.is_sparse, 'A must be sparse'
            assert A.is_coalesced(), 'A must be coalesced'

            out = torch.sum(A, dim=dim).to_dense().unsqueeze(dim=dim)

            if ctx.needs_input_grad[0]:
                ctx.dim = dim if dim >= 0 else ( dim + A.dim() )
                ctx.save_for_backward(A)

            return out

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):        
            dim = ctx.dim
            A, = ctx.saved_tensors
        
            if grad_output.is_sparse:
                assert False, "Didn't handle this case"
                pass
            else:
                inds_out = A.indices().clone()
                inds_out[dim,:] = 0
                
                vals = grad_output[tuple(inds_out)]

                grad_input = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=vals, size=A.shape)

            return grad_input, None


    # Equivalent to: A*B, where A is sparse, B is dense, and the shape of B is broadcastable to A
    class mul_sparse_dense(torch.autograd.Function): # TODO: Try to get rid of coalesce here
        @staticmethod
        def forward(ctx, A, B):
            assert A.is_sparse, 'A must be sparse'
            assert not B.is_sparse, 'B cannot be sparse'
            assert (torch.logical_or(torch.tensor(A.shape) == torch.tensor(B.shape), torch.tensor(B.shape) == 1)).all(), "B must be of size that allows broadcasting to A"
            assert A.is_coalesced(), 'A must be coalesced'

            # Dimensions along B needs to be broadcast to A
            broadcast_dims = tuple(torch.nonzero(torch.tensor(B.shape) == 1))

            inds = A.indices().clone()       
            inds[broadcast_dims, :] = 0
            
            vals = A.values() * B[tuple(inds)]

            out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=vals, size=A.shape)

            A_save = A if ctx.needs_input_grad[1] else None
            B_save = B if ctx.needs_input_grad[0] else None

            ctx.save_for_backward(A_save, B_save)
            ctx.broadcast_dims = broadcast_dims

            return out

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):        
            A, B = ctx.saved_tensors
            broadcast_dims = ctx.broadcast_dims

            out_A = out_B = None

            if ctx.needs_input_grad[0]:
                # The to_sparse() below is just to make sure that B, which is dense, doesn't get broadcast
                # to the size of grad_output; it is the same size of A, which can be huge.

                if True:
                    out_A = B * grad_output.to_sparse()                    
                else:
                    # TODO: Implement this more efficient method. There is a problem with dims due to broadcasting
                    assert_coalesced(grad_output)
                    inds = grad_output.indices()
                    #inds[ctx.broadcast_dims, :] = 0
                    vals = grad_output.values().clone()
                    vals *= B[tuple(inds)]
                    out_A = sp.sparse_coo_tensor_coalesced(indices=inds, values=vals, size=grad_output.shape)
                    
            
            if ctx.needs_input_grad[1]:
                # B is dense, so the gradient with respect to B should also be dense.
                if len(broadcast_dims) > 0:
                    out_B = A*grad_output # This is still sparse and can be huge
                    out_B = out_B.sum(dim=broadcast_dims) #TODO: suspected for slowness

                    # Restore original size as before sum()
                    for i in broadcast_dims:
                        out_B = out_B.unsqueeze(dim=int(i))
                    
                    out_B = out_B.to_dense()
                else:
                    # Note that if B didn't need to be broadcast, this size is not huge
                    out_B = (A*grad_output).to_dense()

            return out_A, out_B


    # Equivalent to: A/B, where A is sparse, B is dense, and the shape of B is broadcastable to A
    class div_sparse_dense(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B):
            assert A.is_sparse, 'A must be sparse'
            assert not B.is_sparse, 'B cannot be sparse'
            assert (torch.logical_or(torch.tensor(A.shape) == torch.tensor(B.shape), torch.tensor(B.shape) == 1)).all(), "B must be of size that allows broadcasting to A"
            assert A.is_coalesced(), 'A must be coalesced'

            if sw_embedding_debug_mode:
                assert (B > 0).all(), 'B cannot contain zeros'

            broadcast_dims = tuple(torch.nonzero(torch.tensor(B.shape) == 1))

            inds = A.indices().clone()       
            inds[broadcast_dims, :] = 0
            
            vals = A.values() / B[tuple(inds)]

            out = sp.sparse_coo_tensor_coalesced(indices=A.indices(), values=vals, size=A.shape)

            A_save = A if ctx.needs_input_grad[1] else None
            B_save = B if True in ctx.needs_input_grad else None

            ctx.save_for_backward(A_save, B_save)
            ctx.broadcast_dims = broadcast_dims

            return out

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):        
            A, B = ctx.saved_tensors
            broadcast_dims = ctx.broadcast_dims

            out_A = out_B = None
        
            if ctx.needs_input_grad[0]:
                #TODO: Why do we need grad_output.to_sparse().coalesce()?
                out_A = sp.div_sparse_dense(grad_output.to_sparse().coalesce(), B) 
            
            if ctx.needs_input_grad[1]:
                # B is dense, so the gradient with respect to B should also be dense.
                if len(broadcast_dims) > 0:
                    out_B = A*grad_output # This is still sparse and can be huge
                    out_B = out_B.sum(dim=broadcast_dims) # TODO: Similarly to mul_sparse_div, suspected for slowness

                    # Restore original size as before sum()
                    for i in broadcast_dims:
                        out_B = out_B.unsqueeze(dim=int(i))
                    
                    out_B = out_B.to_dense()
                    out_B = -out_B / torch.square(B)
                else:
                    out_B = (A*grad_output).to_dense() / (-torch.square(B))
                    assert False, "Did not test this case. Make sure this works."

            return out_A, out_B



    # Equivalent to: A*B, where A and B are sparse and are of the same shape and nonzero pattern
    class mul_sparse(torch.autograd.Function):        
        @staticmethod
        def forward(ctx, A, B):
            assert A.is_sparse and B.is_sparse
            assert (A.shape == B.shape), 'A and B must have the same shape'
            
            assert_coalesced(A)
            assert_coalesced(B)

            if sw_embedding_debug_mode:
                assert (A.indices() == B.indices()).all(), 'A and B nonzero indices do not match'

            inds = A.indices()

            A_vals = A.values()
            B_vals = B.values()

            out_vals = A_vals*B_vals

            A_vals = A_vals if ctx.needs_input_grad[1] else None
            B_vals = B_vals if ctx.needs_input_grad[0] else None

            ctx.save_for_backward(A_vals, B_vals)
            ctx.shape = A.shape
            ctx.inds = inds

            return sp.sparse_coo_tensor_coalesced(indices=inds, values=out_vals, size=A.shape)

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            A_vals, B_vals = ctx.saved_tensors
            shape = ctx.shape
            inds = ctx.inds

            out_A = out_B = None

            if ctx.needs_input_grad[0]:
                out_A = sp.sparse_coo_tensor_coalesced(indices=inds, values=B_vals, size=shape) * grad_output

            if ctx.needs_input_grad[1]:
                out_B = sp.sparse_coo_tensor_coalesced(indices=inds, values=A_vals, size=shape) * grad_output

            return out_A, out_B


    # Equivalent to torch.sinc(A) * torch.cos(B), where A and B are spase tensors of the same shape and nonzero pattern.
    class sinc_cos_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, B):
            assert A.is_sparse and B.is_sparse
            assert (A.shape == B.shape), 'A and B must have the same shape'
            assert A.is_coalesced(), 'A must be coalesced'
            assert B.is_coalesced(), 'B must be coalesced'

            if sw_embedding_debug_mode:
                assert (A.indices() == B.indices()).all(), 'A and B nonzero indices do not match'

            inds = A.indices()

            A_vals = A.values()
            B_vals = B.values()

            A_sinc = torch.sinc(A_vals)
            B_cos = torch.cos(B_vals)

            out_vals = A_sinc*B_cos

            A_vals = A_vals if True in ctx.needs_input_grad else None
            B_vals = B_vals if True in ctx.needs_input_grad else None

            # Note: We could have saved A_sinc and B_cos here for the backward calculation, but these tensors can be huge and cause a memory bottleneck, so better recalculte them
            #       on backward pass. 
            ctx.save_for_backward(A_vals, B_vals)
            ctx.shape = A.shape
            ctx.inds = inds

            return sp.sparse_coo_tensor_coalesced(indices=inds, values=out_vals, size=A.shape)

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            A_vals, B_vals = ctx.saved_tensors
            shape = ctx.shape
            inds = ctx.inds

            out_A = out_B = None

            if ctx.needs_input_grad[0]:
                out_vals_A = sp.dsinc(A_vals) * torch.cos(B_vals)
                out_A = sp.sparse_coo_tensor_coalesced(indices=inds, values=out_vals_A, size=shape) * grad_output
                del out_vals_A

            if ctx.needs_input_grad[1]:
                out_vals_B = torch.sinc(A_vals) * (-torch.sin(B_vals))
                out_B = sp.sparse_coo_tensor_coalesced(indices=inds, values=out_vals_B, size=shape) * grad_output
                del out_vals_B

            return out_A, out_B



    # Equivalent to torch.sort(X, dim=dim, descending=descending)
    class sort(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X, dim, descending):
            Xs, Xi = torch.sort(X, dim=dim, descending=descending)

            # Store the dimension and sorting permutation for back propagation
            if ctx.needs_input_grad[0]:
                ctx.dim = dim if dim >= 0 else ( dim + X.dim() )

                # Try to save space
                Xi_max = X.shape[dim]-1
                #assert Xi_max == Xi.max() # Sanity check

                if Xi_max <= torch.iinfo(torch.int16).max:
                    Xi = Xi.to(dtype=torch.int16)
                elif Xi_max <= torch.iinfo(torch.int32).max:
                    Xi = Xi.to(dtype=torch.int32)

                ctx.save_for_backward(Xi)

            return Xs, Xi

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output, aaa):

            dim = ctx.dim
            Xi, = ctx.saved_tensors

            Xi_inv = torch.argsort(Xi, dim=dim)
            del Xi

            if not ctx.needs_input_grad[0]:
                grad_input = None
            elif grad_output.is_sparse:
                assert False, 'Check this case. Make sure it works.'
                grad_output = grad_output.coalesce()
                inds = grad_output.indices().clone()
                inds[ctx.dim,:] = Xi_inv[tuple(inds)]
                grad_input = sp.sparse_coo_tensor_coalesced(indices=inds, values=grad_output.values(), size=grad_output.shape)
            else:
                grad_input = torch.gather(grad_output, dim=ctx.dim, index=Xi_inv)

            return grad_input, None, None

    # Equivalent to torch.cumsum(X, dim=dim)
    class cumsum_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X, dim):
            assert X.is_coalesced(), 'X must be coalesced'

            ctx.dim = dim if dim >= 0 else ( dim + X.dim() )
            #return sp.sparse_cumsum(X, dim=ctx.dim)
            return sp.sparse_cumsum_alt1(X, dim=ctx.dim)

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):            
            dim = ctx.dim

            if ctx.needs_input_grad[0]:
                G = sp.sparse_flip(grad_output, dim=dim)
                #G = sp.sparse_cumsum(G, dim=dim)
                G = sp.sparse_cumsum_alt1(G, dim=dim)
                grad_input = sp.sparse_flip(G, dim=dim)
            else:
                grad_input = None

            return grad_input, None


    # Replicates the sparse tensor A n times along the dimension dim
    class repmat_sparse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, n, dim):
            ctx.dim = dim if dim >= 0 else ( dim + A.dim() )
            ctx.shape = A.shape

            dim = ctx.dim

            assert_coalesced(A)

            # Three different variants to yield the same result.
            variant = 3 # preferable: no. 3

            # Simplest formulation. Requires a lot of memory.
            if variant == 1:
                return torch.cat([A for _ in range(n)], dim=ctx.dim).coalesce()

            # Memory-efficient, simple formuation, but time-inefficient.
            elif variant == 2:
                out = A
                for i in range(n-1):
                    out = torch.cat([out, A])

            # Memory and time efficient.
            elif variant == 3:
                v = A.shape[dim] * torch.arange(n, device=A.device)
                v = torch.kron(v, torch.ones(A.values().numel(), device=v.device, dtype=v.dtype))

                inds = A.indices().repeat([1,n])                
                inds[dim,:] += v
                del v

                vals = A.values().repeat([n,])

                out_shape = list(A.shape)
                out_shape[dim] = n*out_shape[dim]

                del A

                inds2, vals2 = sp.sort_inds_vals(inds, vals, out_shape, ensure_unique=True)
                del inds, vals

                out = sp.sparse_coo_tensor_coalesced(indices=inds2, values=vals2, size=out_shape)

                return out

            else:
                raise RuntimeError('This should not happen')


            return out

        
        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):

            if ctx.needs_input_grad[0]:
                assert_coalesced(grad_output)
                grad_output = grad_output
                inds = grad_output.indices().clone()
                inds[ctx.dim] = torch.remainder(inds[ctx.dim], ctx.shape[ctx.dim])

                # Here we must coalesce
                grad_input = torch.sparse_coo_tensor(indices=inds, values=grad_output.values(), size=ctx.shape)
                grad_input = grad_input.coalesce()
            else:
                grad_input = None

            return grad_input, None, None




#############################################################################################################
##                                        Sparse tensor operations                                         ##
#############################################################################################################

class sp:
    # Create a COO sparse tensor in a coalesced state, assuming that the input indices are already coalesced.
    # If the command sp.verify_coalescence(out) is not commented, the tensor is verified for being correctly coalesced.
    def sparse_coo_tensor_coalesced(indices, values, size):
        out = torch.sparse_coo_tensor(indices=indices, values=values, size=size, is_coalesced=True)
        debug = sw_embedding_debug_mode

        if debug:
            sp.verify_coalescence(out)

        return out   

    # returns a coalesced copy of A assuming that the indices of A are unique but just possibly unsorted
    def coalesce_unique(A):
        A_shape = A.shape
        inds2, vals2 = sp.sort_inds_vals(indices=A._indices(), values=A._values(), shape=A.shape, ensure_unique=True)
        del A
        return sp.sparse_coo_tensor_coalesced(inds2, vals2, size=A_shape)

    # Verify that a sparse input tensor A is correctly coalesced.
    def verify_coalescence(A):
        assert A.is_coalesced(), 'verify_coalescence: input tensor is not coalesced'
        B = torch.sparse_coo_tensor(indices=A.indices(), values=A.values(), size=A.shape).coalesce()
        assert (B.indices() == A.indices()).all(), 'verify_coalescence: index mismatch in input'
        assert (B.values() == A.values()).all(), 'verify_coalescence: value mismatch in input'


    # The inverse of torch.unravel_index()
    def ravel_index(indices, shape):
        assert indices.dim() == 2, 'indices must be a 2-dimensional tensor'
        nd = indices.shape[0]

        if not isinstance(shape, torch.Tensor):
            shape = torch.tensor(shape, device=indices.device, dtype=indices.dtype)

        weights = shape.reshape([nd,1]).flip(dims=(0,))[0:-1].cumprod(dim=0).flip(dims=(0,))
        weights = torch.cat((weights, torch.ones(size=(1,1), device=weights.device, dtype=weights.dtype)), dim=0)

        out = torch.sum(indices*weights, dim=0) 
        return out


    # Sort indices and values. Similar to coalesce(), but does not assume nor impose uniqueness.
    def sort_inds_vals(indices, values, shape=None, ensure_unique=False):

        if shape is None:
            shape, _ = torch.max(indices, dim=1)
            shape += 1
        elif isinstance(shape, (tuple,list,torch.Size)):
            shape = torch.tensor(shape, device=indices.device, dtype=indices.dtype)
        elif isinstance(shape, torch.Tensor):
            pass
        else:
            raise RuntimeError('sort_inds_vals: invalid shape data type')

        inds1d = sp.ravel_index(indices, shape)
        
        debug = sw_embedding_debug_mode
        if debug:
            assert ( len(torch.unique(inds1d)) == len(inds1d) ), 'indices are not unique'

        _, sort_perm = torch.sort(inds1d)
        del inds1d

        return (indices[:,sort_perm], values[sort_perm])



    # Entrywise division of sparse A by dense A. Supports broadcasting of B to A.
    def div_sparse_dense(A,B): # TODO: Try to get rid of coalesce here
        assert A.is_sparse, 'A must be sparse'
        assert not B.is_sparse, 'B must be dense'
        assert (torch.logical_or(torch.tensor(A.shape) == torch.tensor(B.shape), torch.tensor(B.shape) == 1)).all(), "B must be of size that allows broadcasting to A"
        assert A.is_coalesced(), 'A must be coalesced'
        assert not torch.is_grad_enabled(), 'This function can only be called within torch.no_grad(), as it is not meant to calculate gradients.'

        B = B.expand(A.shape)

        inds = A.indices()
        A_vals = A.values()

        out_vals = A_vals / B[tuple(inds)]

        return sp.sparse_coo_tensor_coalesced(indices=inds, values=out_vals, size=A.shape)


    def cumsum_worker(tensor, output_list, index):
        result = torch.cumsum(tensor, dim=0)        
        output_list[index] = result


    @torch.jit.script
    def cumsum_segments(A, start_inds, end_inds):        
        nSegments = len(start_inds)
        streams = [torch.cuda.Stream() for _ in range(nSegments)]
        out = torch.empty_like(A)
        for i in range(nSegments):
            with torch.cuda.stream(streams[i]):
                #result = torch.cumsum(slice, dim=0)
                #results.append(result)
                out[start_inds[i]:end_inds[i]] = torch.cumsum(A[start_inds[i]:end_inds[i]], dim=0)
        for stream in streams:
            stream.synchronize()
        return out


    #@torch.jit.script
    def sparse_cumsum_alt1(A, dim):
        # Ensure the input is a sparse tensor
        if not A.is_sparse:
            raise ValueError("Input tensor must be a sparse tensor.")
        
        # Coalesce the tensor to ensure unique indices
        # assert_coalesced(A)
        
        inds = A.indices()
        vals = A.values()
        
        # Shape of the sparse tensor
        shape = list(A.shape)
        
        # Get the other dimensions excluding the one we're summing over,
        # and get the shape along these dimensions.
        dims2 = [d for d in range(len(shape)) if d != dim]
        shape2 = [shape[d] for d in range(len(shape)) if d != dim]
        #dims2 = list(range(len(shape))).remove(dim)
        #shape2 = [shape[i] for i in dims2]
        
        # Get the unique keys for the other dimensions
        keys = sp.ravel_index(inds[dims2,:], shape2)

        # Sort the keys and get the counts of each unique key
        keys_sorted, sort_inds = torch.sort(keys, dim=0, stable=True)
        _, counts = torch.unique_consecutive(keys_sorted, return_counts=True)

        start_inds = torch.cumsum(counts, dim=0)        
        start_inds = torch.cat( (torch.zeros(1, device=start_inds.device, dtype=start_inds.dtype), start_inds[0:-1]), dim=0)
        end_inds = start_inds + counts
        
        # Sort the values and split them according to the keys at the corrsponding indices
        vals_sorted = vals[sort_inds]
        vals_split = torch.split(vals_sorted, list(counts), dim=0)

        if True:       
            vals_sorted_cumsum = sp.cumsum_segments(vals_sorted, start_inds, end_inds)
        else:
            s = time.time()
            streams = [torch.cuda.Stream() for _ in range(len(vals_split))]
            t = time.time()-s
            #print('Stream creation time: ', t)

            # Asynchronously apply the cumsum function to each slice using its own stream
            s = time.time()
            vals_sorted_cumsum = torch.empty_like(vals_sorted)
            #results = []
            for i, slice in enumerate(vals_split):
                with torch.cuda.stream(streams[i]):
                    #result = torch.cumsum(slice, dim=0)
                    #results.append(result)
                    vals_sorted_cumsum[start_inds[i]:end_inds[i]] = torch.cumsum(slice, dim=0)
            t = time.time()-s
            #print('Result calculation time: ', t)

            # Wait for all streams to complete
            s = time.time()
            for stream in streams:
                stream.synchronize()
            t = time.time()-s
            #print('Synchronize time: ', t)

            s = time.time()
            #vals_sorted_cumsum = torch.cat(results, dim=0)
            t = time.time()-s
            #print('cat time: ', t)

        perm_inv = torch.argsort(sort_inds, dim=0)
        vals_out = vals_sorted_cumsum[perm_inv]
        
        # Create a new sparse tensor with cumulative sum values
        out = sp.sparse_coo_tensor_coalesced(indices=inds, values=vals_out, size=shape)
        
        return out
        
    
    def sparse_cumsum(A, dim):
        assert A.is_sparse, 'A must be sparse'
        assert not torch.is_grad_enabled(), 'This function can only be called within torch.no_grad(), as it is not meant to calculate gradients.'

        dim = dim if dim >= 0 else ( dim + A.dim() )

        if dim != A.dim()- 1:
            #print('A device: %s  A mmz: %d' % (A.device, A.values().numel()))

            # Two equivalent implementations. The first is clearer, the second is more memory-efficient.
            if False:
                return sp.sparse_cumsum(  A.transpose(-1,dim).coalesce()  ,  -1).transpose(-1,dim).coalesce()
            else:
                A = A.transpose(-1,dim)
                A = sp.coalesce_unique(A)
                out = sp.sparse_cumsum(A, -1)
                del A
                out = out.transpose(-1,dim)
                out = sp.coalesce_unique(out)

            return out

        assert A.is_coalesced(), 'A must be coalesced'

        # Note: We convert here to float64 because on float32 the numerical error is accumulated over the whole of A and can be quite high
        # when working with float32 (rel. err ~1e-3).
        # TODO: We will not need this conversion when torch adds support for cumsum with a reset mask, 
        #       or a designated cumsum for sparse matrices.
        #       On top of the high loss of significant digits, this function is also a memory bottleneck.
        #       Better replace it as soon as possible.

        A_shape = A.shape

        orig_dtype = A.dtype
        A = A.to(dtype=torch.float64)

        # Variant 1 is simpler to read.
        # Not sure which of the two is faster or more memory efficient.
        # Possibly variant 1 since it avoids coalesce.
        variant = 1 

        if variant == 1:
            sums = A.sum(dim=-1)
            assert_coalesced(sums)
            sums = sums = ag.unsqueeze_sparse.apply(sums,-1)

        elif variant == 2:
            sums_shape = list(A.shape)
            sums_shape = sums_shape[0:-1]
            sums_shape = tuple(sums_shape)

            # Indices are not coalesced            
            sums = torch.sparse_coo_tensor(indices=A.indices()[0:-1, :], values=A.values(), size=sums_shape)
            sums = sums.coalesce()

            sums = ag.unsqueeze_sparse.apply(sums,-1)            

        elif variant == 3:
            # If we had a good function for sparse tensor multiplication across arbitrary dimension, we could calculate
            # sums by multiplying A with a vector of ones. Not sure if it would be better than variant 1 though.
            pass

        
        else:
            raise RuntimeError('This should not happen')

        # Variant 1 is the simplest to read, but it requires more memory
        # Variant 2 is more memory efficient; it is similar to variant 1, but avoids coalesce
        #   Note that it makes use of the PyTorch's internal functions _indices() and _values()
        # Variant 3 is the most low-level method. Not sure if it is better than 2.
        variant = 2   # preferable is no. 2, unless something goes wrong due to using _indices() and _values().
                      # in this case, no. 3 is preferable.

        if variant == 1:
            A_walled = torch.cat((A, -sums), dim=-1)
            del A, sums
            A_walled = A_walled.coalesce()

        elif variant == 2:
            A_walled = torch.cat((A, -sums), dim=-1)
            del A, sums
            inds2, vals2 = sp.sort_inds_vals(A_walled._indices(), A_walled._values(), shape=A_walled.shape, ensure_unique=True)
            walled_shape = A_walled.shape
            del A_walled
            A_walled = sp.sparse_coo_tensor_coalesced(indices=inds2, values=vals2, size=walled_shape)

        elif variant == 3:
            sums_inds = sums.indices()
            sums_inds[-1,:] += A.shape[-1]
            sums_values = sums.values()
            del sums

            inds_walled = torch.cat( (A.indices(), sums_inds), dim=1)                
            inds_walled[len(A.indices()):, -1] = A.shape[-1]
            del sums_inds

            vals_walled = torch.cat( (A.values(), -sums_values), dim=0)
            del sums_values

            walled_shape = list(A.shape)
            walled_shape[-1] = walled_shape[-1]+1
            walled_shape = tuple(walled_shape)

            # Indices are not coalesced
            inds2, vals2 = sp.sort_inds_vals(inds_walled, vals_walled, shape=walled_shape, ensure_unique=True)
            del A, inds_walled, vals_walled
            A_walled = sp.sparse_coo_tensor_coalesced(indices=inds2, values=vals2, size=walled_shape)
            del inds2, vals2

        else:
            raise RuntimeError('This should not happen')

        inds = A_walled.indices()
        vals = A_walled.values().cumsum(dim=0)
        del A_walled

        subset = (inds[-1,:] < A_shape[-1])

        out = sp.sparse_coo_tensor_coalesced(indices=inds[:,subset], values=vals[subset], size=A_shape)
        del inds, vals, subset

        out = out.to(dtype=orig_dtype)
        assert_coalesced(out)

        return out


    def sparse_flip(A, dim):
        assert not torch.is_grad_enabled(), 'This function can only be called within torch.no_grad(), as it is not meant to calculate gradients.'
        assert A.is_sparse
        assert A.is_coalesced()

        inds = A.indices().clone()
        vals = A.values()

        inds[dim, :] = A.shape[dim] - inds[dim, :] - 1

        inds2, vals2 = sp.sort_inds_vals(inds, vals, shape=A.shape, ensure_unique=True)
        out = sp.sparse_coo_tensor_coalesced(indices=inds2, values=vals2, size=A.shape)

        return out


    def permute(A, dim, perms, broadcast_perms_dim=None, backward_mode=False):
        assert not torch.is_grad_enabled(), 'This function can only be called within torch.no_grad(), as it is not meant to calculate gradients.'
        if broadcast_perms_dim is None:
            assert A.shape == perms.shape
        else:
            assert replace_in_tuple(tuple(A.shape), broadcast_perms_dim, 1) == perms.shape

        # Invert the given permutations.
        # Although the second option has linear time complexity, the first one is simpler and there is no difference in the actual running times, even on huge inputs.
        if backward_mode:
            # If we're in backward mode, do not invert
            perm_invs = perms
        elif False:
            perm_invs = torch.argsort(perms, dim=dim)
        else:
            perm_invs = torch.empty_like(perms)
            ar = torch.arange(perms.shape[dim], dtype=perms.dtype, device=perms.device)
            ar = ar.reshape((1,)*dim + (len(ar),) + (1,)*(len(perms.shape)-(dim+1))).expand_as(perms)
            perms = perms.to(dtype=torch.int64)
            perm_invs.scatter_(dim=dim, index=perms, src=ar)
            del ar

        A = sp.coalesce_unique(A) # TODO: Tough to get rid of when permute() is applied to grad_out
        inds = A.indices().clone()
            
        if broadcast_perms_dim is not None:
            perm_inds = inds.clone()
            perm_inds[broadcast_perms_dim,:] = 0 
        else:
            perm_inds = inds

        inds[dim,:] = perm_invs[tuple(perm_inds)]

        inds2, vals2 = sp.sort_inds_vals(inds, A.values().clone(), shape=A.shape, ensure_unique=True)
        del inds, perm_inds, perms, perm_invs

        out = sp.sparse_coo_tensor_coalesced(indices=inds2, values=vals2, size=A.shape)

        return out


    # Returns a tensor of the same size as x, containing the values of d/dx sinc(x)
    def dsinc(x):

        with torch.enable_grad():
            x2 = x.clone().detach()
            x2.requires_grad = True
            y = torch.sinc(x2)
            dy = torch.autograd.grad(y, x2, torch.ones_like(y), create_graph=False)[0]

        return dy



#############################################################################################################
##                                      Mutual Coherence Minimization                                      ##
#############################################################################################################


def minimize_mutual_coherence(X_init, report=True):
    step_size_init = 2000
    nIter_max = 1000
    improvement_thresh = 1e-4 # Use 1e-6 for more thoroughness

    p_vals = [3,6,10,20,50,100,200,500,1000,2000,5000, 1e4, 2e4, 5e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13]

    step_size_curr = step_size_init
    
    n = X_init.shape[0]
    X_curr = X_init
    X_curr = nn.functional.normalize(X_curr, p=2, dim=1, eps=0)
    mu_init = calc_mu_from_G(calc_G(X_curr))
    
    for ip, p_curr in enumerate(p_vals):
        qprintln(report, '\n=== Optimizing with p = %g (%d/%d) ===' % (p_curr, ip+1, len(p_vals)) )
        X_curr, step_size_curr = minimize_mutual_coherence_p(X_curr, p_curr, step_size_init=step_size_curr, improvement_thresh=improvement_thresh, nIter_max=nIter_max, report=report)

        mu_curr = mu_init = calc_mu_from_G(calc_G(X_curr))
        #qprintln('Relative improvement vs. init: %g' % ((mu_init-mu_curr)/(1-mu_init)))

        qprintln(report, '\nIncoherence: %g  Min. pairwise dist: %g' % ( 1.0-mu_curr, np.sqrt(np.maximum(0,2*(1.0-mu_curr.cpu()))) ) )
    return X_curr



def minimize_mutual_coherence_p(X_init, p, step_size_init, improvement_thresh, nIter_max, report):
    # Note: X_init must be normalized to unit rows

    p = np.double(p)

    # Parameters
    step_size_min = 1e-5
    step_size_max = 1e10
    max_num_low_improvements = 5
    step_decrease_factor = 0.5

    assert p >= 2, "p must be greater or equal to 2"
    assert step_size_init >= step_size_min, "Initial step size below minium"
    assert step_size_init <= step_size_max, "Initial step size above maximum"

    # Initialization
    n = X_init.shape[0]

    onevec = torch.ones([n,1], dtype=X_init.dtype, device=X_init.device)

    mu_init = calc_mu_from_G(calc_G(X_init))

    ## Initialize first iteration
    step_size_curr = step_size_init

    # The first step size is chosen as follows: Start from step_size_init.
    # If at the first iteration this step size yields an objective decrease,
    # iteratively increase the step size until the objective increases.
    # Then choose the best among the tested step sizes and use it. This
    # prevents getting stuck with too low a step size throughout the
    # optimization process.
    step_size_init_best = step_size_init
    obj_best_at_step_init_seek = np.inf
    finished_step_size_init = False

    low_improvement_counter = 0

    # These four have to be maintained together and consistent with each
    # other.
    X_curr = X_init
    G_curr = calc_G(X_curr) # The Gram matrix of X with its main diagonal annihilated.
    mu_curr, obj_curr = eval_G(G_curr, p)

    qprintln(report, '#%.2d:  Objective: %g  Surrogate incoherence: %g  Step size: %g' % (0, obj_curr, 1.0-obj_curr, step_size_curr) ) 

    rho = np.power( 1.0/(2.0*n*(n-1.0)), 1.0/p )

    for i in range(1, nIter_max+2):
        if i > nIter_max:
            qprintln(report, 'Reached maximal number of iterations. Breaking.')
            break

        # Calculate gradient at current solution
        # For numerical safety, G is normalized so that its
        # largest-magnitude entry equals 1.
        G_normalized = G_curr / mu_curr
        sum_offdiags_norm = torch.sum(torch.pow(torch.abs(G_normalized), p))
        
        grad_curr = rho / torch.pow(sum_offdiags_norm, 1.0 - 1.0/p) * ( ( torch.pow(torch.abs( G_normalized ), p-1.0) * torch.sign(G_normalized) ) @ X_curr - ( torch.pow(torch.abs(G_normalized), p) @ (mu_curr*onevec) ) * X_curr )

        # Calculate and evaluate new candidate solution
        X_new = nn.functional.normalize(X_curr - step_size_curr * grad_curr, p=2, dim=1, eps=0)
        G_new = calc_G(X_new)
        mu_new, obj_new = eval_G(G_new, p)

        # If the objective does not improve
        if obj_new >= obj_curr:
            if finished_step_size_init:
                # Decrease step size                
                if step_size_curr * step_decrease_factor < step_size_min:
                    qprintln(report, '#%.2d:  Objective does not improve at minimal step size. Breaking.' % (i) )
                    break

                step_size_curr = step_size_curr * step_decrease_factor
            else:
                # If we're still seeking the first step size, stop and pick
                # the best step size so far.
                step_size_curr = step_size_init_best
                finished_step_size_init = True
            
            qprintln(report, '#%.2d:  Decreaseing step size: %g' % (i, step_size_curr) )
            continue

        # --> If we're here, the objective has improved.
        
        if not finished_step_size_init:
            if (obj_new < obj_best_at_step_init_seek) and (step_size_curr / step_decrease_factor <= step_size_max):
                obj_best_at_step_init_seek = obj_new
                step_size_init_best = step_size_curr
                step_size_curr = step_size_curr / step_decrease_factor

                # Save the new best solution for later backtracking
                X_stepinit = X_new
                G_stepinit = G_new
                obj_stepinit = obj_new
                mu_stepinit = mu_new

                qprintln(report, '#%.2d:  Trying larger step size: %g' % (i, step_size_curr) )
                continue
            else:
                # If we're seeking the best first step, on the first time
                # that increasing the step does not improve the objective,
                # stop increasing and backtrack to the best solution so
                # far.
                step_size_curr = step_size_init_best
                finished_step_size_init = True

                # Backtrack to best candidate solution
                X_new = X_stepinit
                G_new = G_stepinit
                obj_new = obj_stepinit
                mu_new = mu_stepinit

        # --> If we're here, we accept the candidate solution.

        # We divide by 1-obj_curr rather than obj_curr because this measure
        # is more informative at values near 1.
        improvement_curr = (obj_curr-obj_new) / (1.0-obj_curr)

        # Accept candidate solution        
        X_curr = X_new
        G_curr = G_new
        obj_curr = obj_new
        mu_curr = mu_new
        
        qprint(report, '#%.2d:  Objective: %g  Surrogate incoherence: %g  Step size: %g  Improvement: %g  ' % (i, obj_curr, 1.0-obj_curr, step_size_curr, improvement_curr) )

        if improvement_curr <= improvement_thresh:
            low_improvement_counter =  low_improvement_counter + 1
            qprint(report, 'Low improvement strike %d' % (low_improvement_counter) )

            if low_improvement_counter >= max_num_low_improvements:
                qprintln(report)
                qprintln(report)
                qprintln(report, 'Reached maximal number of consecutive iterations with low improvement. Breaking.')
                break
        else:
            low_improvement_counter = 0

        qprintln(report)

    #qprintln(report, '\nIncoherence: %g' % (1-mu_curr) )

    if mu_curr < mu_init:
        qprintln(report, '\nRelative improvement with p=%g: %g' % (p, (mu_init-mu_curr)/(1.0-mu_init)) )
        X_out = X_curr
        step_size_out = step_size_curr
    else:
        qprintln(report, '\nIncoherence did not improve with p=%g. Reverting to previous solution.' % (p) )
        X_out = X_init
        step_size_out = step_size_init

    return X_out, step_size_out


def calc_G(X):    
    n = X.shape[0]
    G = X@X.transpose(0,1)
    G[range(n),range(n)] = 0
    return G


def calc_mu_from_G(G):
    return torch.max( torch.abs(G)  )


def eval_G(G, p):
    n = G.shape[0]
    mu = calc_mu_from_G(G)
    rho = 1.0/(2.0*n*(n-1.0))
    objective = mu * torch.pow( rho * torch.sum( torch.pow( torch.abs(G/mu), p ) ), 1.0/p )

    return mu, objective



