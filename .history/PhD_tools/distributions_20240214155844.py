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

