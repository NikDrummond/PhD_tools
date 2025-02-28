import numpy as np

def edf(data, alpha=.05, x0=None, x1=None , bins = 100):
    """
    Calculate the empirical distribution function and confidence intervals around it.

    Parameters
    ----------
    data:

    alpha:

    x0:

    x1:

    Returns
    -------
    x:

    y:

    l:

    u:
    """


    x0 = data.min() if x0 is None else x0
    x1 = data.max() if x1 is None else x1
    x = np.linspace(x0, x1, bins)
    N = data.size
    y = np.zeros_like(x)
    l = np.zeros_like(x)
    u = np.zeros_like(x)
    e = np.sqrt(1.0/(2*N) * np.log(2./alpha))
    for i, xx in enumerate(x):
        y[i] = np.sum(data <= xx)/N
        l[i] = np.maximum( y[i] - e, 0 )
        u[i] = np.minimum( y[i] + e, 1 )
    return x, y, l, u

def epdf(data, alpha=0.05, x0=None, x1=None, bins=None, n_bootstrap=1000):
    """
    Estimate the empirical probability density function (EPDF) and confidence intervals.

    Parameters
    ----------
    data : np.ndarray
        1D array of sample data.
    alpha : float, optional
        Confidence level (default is 0.05 for 95% confidence intervals).
    x0 : float, optional
        Lower bound of the x-axis range (default: min(data)).
    x1 : float, optional
        Upper bound of the x-axis range (default: max(data)).
    bins : int, optional
        Number of bins (default: automatic selection using Freedman-Diaconis rule).
    n_bootstrap : int, optional
        Number of bootstrap resamples (default is 1000).

    Returns
    -------
    x : np.ndarray
        Bin centers where the density is evaluated.
    y : np.ndarray
        Estimated density values.
    l : np.ndarray
        Lower bound of the confidence interval.
    u : np.ndarray
        Upper bound of the confidence interval.
    """

    # Validate input
    assert isinstance(data, np.ndarray) and data.ndim == 1, "data must be a 1D NumPy array"
    assert 0 < alpha < 1, "alpha must be in (0,1)"

    # Define x0 and x1 with optional buffer
    x0 = np.min(data) if x0 is None else x0
    x1 = np.max(data) if x1 is None else x1
    buffer = 0.05 * (x1 - x0)  # Small buffer outside observed range
    x0, x1 = x0 - buffer, x1 + buffer

    # Determine number of bins
    if bins is None:
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / np.cbrt(len(data))  # Freedman-Diaconis rule
        bins = max(10, int((x1 - x0) / bin_width))  # Ensure at least 10 bins

    # Compute histogram (EPDF)
    bin_edges = np.linspace(x0, x1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_values, _ = np.histogram(data, bins=bin_edges, density=True)

    # Bootstrap confidence intervals
    boot_samples = np.zeros((n_bootstrap, bins))
    for i in range(n_bootstrap):
        resampled_data = np.random.choice(data, size=len(data), replace=True)
        boot_samples[i], _ = np.histogram(resampled_data, bins=bin_edges, density=True)

    # Compute confidence intervals
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(boot_samples, 100 * (1 - alpha / 2), axis=0)

    return bin_centers, hist_values, lower_bound, upper_bound
# PERT and distance functions

#### NOTE PDF DOES NOT RETURN A PROBABILITY - WILL OFTEN BE GREATER THAN 1

class PERT:
    """PERT probability distribution.

    Implements the PERT (Program Evaluation and Review Technique) probability distribution. 
    Allows specifying the min, mode, and max values to define the distribution.

    Attributes:
        a: Minimum value
        b: Mode value 
        c: Maximum value
        lamb: Shape parameter
        alpha: PERT distribution alpha parameter
        beta: PERT distribution beta parameter 
        mean: Mean of distribution
        var: Variance of distribution

    Methods:
        build: Calculates PERT distribution parameters
        median: Returns median of distribution
        pdf: Evaluates the probability density function    
    """
    def __init__(self, min_val:float, mode:float, max_val:float, lamb:float = 4.0):
        self.a = min_val
        self.b = mode
        self.c = max_val
        self.lamb = lamb

        # some checks
        assert lamb > 0, 'lamb parameter should be greater than 0.'
        assert self.b > self.a, 'minimum value must be lower than mode.'
        assert self.c > self.b, 'maximum values must be greater than mode.'
        assert not isclose(self.a,self.b) | isclose(self.b,self.c) | isclose(self.a,self.c), 'minimum, mode, and maximum values should be different.'

        self.build()
        
    def build(self):

                
        self.alpha = 1 + (self.lamb * ((self.b-self.a) / (self.c-self.a)))
        self.beta = 1 + (self.lamb * ((self.c-self.b) / (self.c-self.a)))
        
        self.mean = (self.a + (self.lamb*self.b) + self.c) / (2+self.lamb)
        self.var = ((self.mean-self.a) * (self.c-self.mean)) / (self.lamb+3)
        
    @property
    def range(self):
        """ Calculates the min-max range
        
        Returns
        -------
        Array:
            Array of range values of max-min.
        """
        return np.asarray(self.c - self.a)
    
    def median(self) -> float:
        median = (beta_dist(self.alpha, self.beta).median() * self.range) + self.a
        return median
    
    def std(self) -> float:
        return np.sqrt(self.var)
    
    def pdf(self, val:np.ndarray) -> np.ndarray:
        x = ((val - self.a) / self.range).clip(0,1)
        pdf_val = beta_dist.pdf(x, self.alpha, self.beta) / self.range
        return pdf_val


def mahalanobis_distance(x:float, expectancy:float, cov:float, absolute:bool = False) -> float:
    """Calculates the Mahalanobis distance between a vector x and a distribution
    with mean expectancy and covariance matrix cov. 

    NOTE THIS IS WRONG - NEED TO FIX BUT NOT CURRENTLY USING IT.

    Parameters:
        x (np.ndarray): Vector to calculate distance for
        expectancy (np.ndarray): Mean of distribution
        cov (np.ndarray): Covariance matrix of distribution
        abs (bool): Whether to take absolute value of distance

    Returns:
        dist (np.ndarray): Mahalanobis distance 
    """
    dist = (x - expectancy) / cov
    if absolute:
        dist = abs(dist)
    return dist 

## Generalised Histogram thresholding


# If this code is shared with paper publication, add that this essentially copied from GHT paper

# cumulative sum
csum = lambda z:np.cumsum(z)[:-1]
#de-cumulative sum
dsum = lambda z:np.cumsum(z[::-1])[-2::-1]

# Use the mean for ties .
argmax = lambda x , f:np.mean(x[:-1][f==np.max(f)]) 
clip = lambda z:np.maximum(1e-30,z)

def image_hist(N, range = None):
    """ Generate bins and counts """
    image = N.ravel().astype(int)
    n = np.bincount(image,minlength = np.max(image) - np.min(image) + 1)
    if range != None:
        if len(range) != 2:
            raise TypeError('Range is not length 2')
        else:
            x = np.arange(range[0], range[1])    
    elif range == None:
        x = np.arange(0,np.max(image) + 1)
    else:
        raise TypeError("range input type not supported")
    
    return n,x

def preliminaries(n, x):
    """ Some math that is shared across each algorithm - refering to GHT, weighted percentile and Otsu"""
    assert np . all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np . all(x[1:] >= x[: -1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0/(w0 + w1)
    p1 = w1/(w0 + w1)
    mu0 = csum(n*x)/w0
    mu1 = dsum(n*x)/w1
    d0 = csum(n*x**2)-w0*mu0**2
    d1 = dsum(n*x**2)-w1*mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1


def GHT(n, x, nu=None, tau=0.1, kappa=0.1, omega=0.5):
    """ GHT implementation"""
    if nu == None:
        nu = abs(len(x)/2)
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
    v0 = clip((p0*nu*tau**2+d0)/(p0*nu+w0))
    v1 = clip((p1*nu*tau**2+d1)/(p1*nu+w1))
    f0 = - d0/v0-w0*np.log(v0) + 2*(w0+kappa*omega)*np.log(w0)
    f1 = - d1/v1-w1*np.log(v1)+2*\
        (w1+kappa*(1-omega))*np.log(w1)
    return argmax(x, f0+f1), f0+f1

###



