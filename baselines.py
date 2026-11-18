"""Analytic moments and report-level quantization used in the evaluation."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from mechanisms.gpm import GPM


def stoch_variance(y,levels):
    y=np.asarray(y);levels=np.asarray(levels)
    if y.min()<levels[0]-1e-10 or y.max()>levels[-1]+1e-10:raise ValueError('quantizer does not cover support')
    j=np.clip(np.searchsorted(levels,y,side='right')-1,0,len(levels)-2)
    return np.maximum(0,(y-levels[j])*(levels[j+1]-y))


def piecewise_max(fn,knots):
    """Exact quadratic-piece maximum via local quadratic interpolation (float)."""
    x=np.unique(np.clip(np.r_[-1,knots,1],-1,1)); a,b=x[:-1],x[1:]
    v0,vm,v1=[np.asarray(fn(z)) for z in (a,(a+b)/2,b)]
    A=2*(v1+v0-2*vm);B=v1-v0-A
    t=np.divide(-B,2*A,out=np.zeros_like(A),where=abs(A)>1e-25)
    t=np.clip(t,0,1)
    return float(max(np.max(v0),np.max(v1),np.max(A*t*t+B*t+v0)))


def tune_levels(lo,hi,bits,objective):
    """Data-free search: fixed extreme levels + equally spaced interior levels.

    Includes the full-range uniform quantizer. This is a small one-dimensional
    postprocessing search, not a certified bit-constrained mechanism optimizer.
    """
    K=2**bits;center=(lo+hi)/2;radius=(hi-lo)/2
    def levels(t):return np.r_[lo,np.linspace(center-radius*t,center+radius*t,K-2),hi]
    uniform_t=(K-3)/(K-1)
    ts=np.unique(np.r_[np.geomspace(.001,.999999,20),uniform_t])
    values=[objective(levels(t)) for t in ts];j=int(np.argmin(values))
    best_t=ts[j];best=values[j]
    if 0<j<len(ts)-1:
        r=minimize_scalar(lambda t:objective(levels(t)),bounds=(ts[j-1],ts[j+1]),method='bounded',options={'xatol':1e-5,'maxiter':35})
        if r.success and r.fun<best:best_t=float(r.x)
    return levels(best_t)


class Piecewise:
    """Low density over [-A,A], elevated density over [s*x-h,s*x+h]."""
    def __init__(self,A,s,h,low,high,source):
        self.A,self.s,self.h,self.low,self.high=A,s,h,low,high
        self.source=source;self.N=None
        x=np.array([-1.,-.37,0,.61,1])
        if abs(2*A*low+2*h*(high-low)-1)>1e-10:raise ValueError('normalization')
        if s+h>A+1e-9:raise ValueError('interval outside support')
        np.testing.assert_allclose(self.moments(x)[0],x,atol=1e-9,rtol=1e-9)
    def moments(self,x,bits=None,levels=None):
        x=np.asarray(x,float);l=self.s*x-self.h;r=self.s*x+self.h
        mean=(self.high-self.low)*(r*r-l*l)/2
        if bits is None and levels is None:
            second=self.low*2*self.A**3/3+(self.high-self.low)*(r**3-l**3)/3
        else:
            if levels is None:levels=np.linspace(-self.A,self.A,2**bits)
            seconds=levels**2
            cumulative=np.r_[0,np.cumsum(np.diff(levels)*(seconds[:-1]+seconds[1:])/2)]
            def primitive(z):
                j=np.clip(np.searchsorted(levels,z,side='right')-1,0,len(levels)-2)
                d=z-levels[j];slope=(seconds[j+1]-seconds[j])/(levels[j+1]-levels[j])
                return cumulative[j]+seconds[j]*d+slope*d*d/2
            second=self.low*cumulative[-1]+(self.high-self.low)*(primitive(r)-primitive(l))
        return mean,second
    def worst(self):return piecewise_max(lambda x:self.moments(x)[1]-x*x,[-1,0,1])
    def tuned(self,bits):
        def objective(levels):
            knots=np.r_[(levels+self.h)/self.s,(levels-self.h)/self.s]
            return piecewise_max(lambda x:self.moments(x,levels=levels)[1]-x*x,knots)
        return tune_levels(-self.A,self.A,bits,objective)


def pm_variance(e,t,x):
    E=np.exp(e)
    return (t+1)/(E-1)*np.asarray(x)**2+(t+E)*((t+1)**3+E-1)/(3*t*t*(E-1)**2)


def pm_opt_t(e):
    """Zhao et al., arXiv:2004.08856v2, Lemma 7 / Eqs. (30)-(32).

    Numerically evaluate the same unique optimum as the radical expression (30).
    With E=exp(e), the derivative numerator is
        t^4 + 2 E t^3 - 2 E t - E^2.
    V''(t)=2*(t^4+4Et+3E^2)/(3*t^4*(E-1)^2)>0.
    For e>0 the root lies in (1,E). No fragile branch at e=log(sqrt(2)).
    """
    E=float(np.exp(e))
    if not np.isfinite(E) or E<=1:raise ValueError('PM-OPT requires positive finite epsilon')
    return brentq(lambda t:t**4+2*E*t**3-2*E*t-E**2,1.,E,xtol=1e-14)


def continuous(name,e):
    E=np.exp(e)
    if name in ('PM','PM-SUB','PM-opt'):
        t=np.exp(e/(2 if name=='PM' else 3))
        if name=='PM-opt':
            t=pm_opt_t(e)
        h=(E+t)/(t*(E-1));s=h*t;A=s+h
        low=t*(E-1)/(2*(t+E)**2)
        m=Piecewise(A,s,h,low,E*low,dict(implementation='TOPM PM family',t=t,design='min worst variance' if name=='PM-opt' else name))
        if name=='PM-opt':m.source['reference']='arXiv:2004.08856v2, Algorithm 3, Eqs. (26)-(32); unique derivative root equivalent to Eq. (30)'
        np.testing.assert_allclose(m.moments(np.array([-1.,0,1]))[1]-np.array([1.,0,1]),pm_variance(e,t,np.array([-1.,0,1])),rtol=1e-10)
        return m
    if name=='OGPM-U':
        g=GPM(1,e,np.random.default_rng(0),range=1,k=1)
        m=Piecewise(g.A,(g.C+1)/2,(g.C-1)*(2*g.C+1)/(2*g.C),g.low_density_01/2,g.high_density_01/2,
                    dict(implementation='mechanisms.gpm.GPM',mapping='x=2u-1,y=2z-1; variance[-1,1]=4 variance[0,1]'))
        x=np.linspace(-1,1,9);np.testing.assert_allclose(m.moments(x)[1]-x*x,g.squared_error(x),rtol=1e-10)
        return m
    if name=='SW':
        b=(e*E-E+1)/(2*E*(E-1-e));q=1/(2*E*b+1);p=E*q;slope=2*b*(p-q)
        return Piecewise((1+2*b)/slope,1/slope,2*b/slope,q*slope/2,p*slope/2,
                         dict(implementation='SW affine unbiased decoder',slope=slope))
    raise ValueError(name)


def categorical_moments(name,e,x,categories,rounding='stochastic'):
    """Direct linear DE/OUE mean decoder; no nonlinear histogram projection.

    Stochastic input rounding to endpoint-inclusive nodes preserves E[Z|x]=x.
    Conditional category moments include the rounding variance analytically.
    """
    nodes=np.linspace(-1,1,categories);E=np.exp(e)
    if rounding=='stochastic':
        mean=np.asarray(x,float);cat_second=mean**2+stoch_variance(mean,nodes)
    else:
        raise ValueError(rounding)
    if name=='DE':
        q=1/(E+categories-1);p=E*q;delta=p-q
        second=(q*np.sum(nodes**2)+delta*cat_second)/(delta**2)
    else:
        p=.5;q=1/(E+1);delta=p-q
        # Conditional bit independence applies given the category. After input
        # rounding use total variance; bits are not marginally independent.
        variance_given=(q*(1-q)*np.sum(nodes**2)+(p*(1-p)-q*(1-q))*cat_second)/delta**2
        second=variance_given+cat_second
    return mean,second
