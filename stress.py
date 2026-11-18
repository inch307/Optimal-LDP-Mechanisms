"""Boundary-safe stress distributions and piecewise moment integration."""
import numpy as np


class Moments:
    def __init__(self,spec):
        self.spec=spec;self.knots=np.asarray(spec.get('grid',[-1,1]),float)
    def __call__(self,x):
        x=np.asarray(x,float)
        if self.spec['kind']=='quadratic':return x,np.polynomial.polynomial.polyval(x,self.spec['coefficients'])
        return np.interp(x,self.knots,self.spec.get('mean',self.knots)),np.interp(x,self.knots,self.spec['second'])
    def variance(self,x):
        mu,sec=self(x);return sec-mu*mu
    def worst(self):
        a,b=self.knots[:-1],self.knots[1:];v0,vm,v1=[self.variance(x) for x in [a,(a+b)/2,b]]
        A=2*(v1+v0-2*vm);B=v1-v0-A
        t=np.divide(-B,2*A,out=np.zeros_like(A),where=abs(A)>1e-25);t=np.clip(t,0,1)
        points=np.r_[a,b,a+(b-a)*t];values=self.variance(points);j=int(np.argmax(values))
        return float(points[j]),float(values[j])


def neighborhood(center,d,symmetric=True):
    """Uniform distance R~U[0,2d], equal feasible direction probabilities.

    Returns uniform/point components (left,right,mass). No clipping/rejection.
    E[distance to the selected +/- center]=d. Near overlapping mirrored centers
    this need not equal distance to the nearest member of the worst-point set.
    """
    if not -1<=center<=1 or not 0<=d<=.5:raise ValueError('center in [-1,1], d in [0,.5] required')
    if d==0:base=[(center,center,1.)]
    else:
        h=2*d;a=min(h,1-center,1+center);base=[]
        if a>0:base.extend([(center-a,center,a/(2*h)),(center,center+a,a/(2*h))])
        if h>a:
            base.append((center-h,center-a,(h-a)/h) if center>=0 else (center+a,center+h,(h-a)/h))
    if not symmetric:return base
    return [(a,b,p/2) for a,b,p in base]+[(-b,-a,p/2) for a,b,p in base]


def integrate(fn,knots,components):
    """Three-point Gauss integration after splitting at all moment knots."""
    nodes,weights=np.polynomial.legendre.leggauss(3);answer=0.
    for a,b,mass in components:
        if a==b:answer+=mass*float(np.asarray(fn(np.array([a])))[0]);continue
        cuts=np.unique(np.r_[a,knots[(knots>a)&(knots<b)],b])
        lo,hi=cuts[:-1],cuts[1:]
        x=((lo+hi)[:,None]/2+(hi-lo)[:,None]/2*nodes).ravel()
        value=np.asarray(fn(x)).reshape(-1,3)@weights
        answer+=mass/(b-a)*float(((hi-lo)/2)@value)
    return answer


def expected(m,components,n):
    bias=integrate(lambda x:m(x)[0]-x,m.knots,components)
    variance=integrate(m.variance,m.knots,components)
    risk=integrate(lambda x:m(x)[1]-2*x*m(x)[0]+x*x,m.knots,components)
    # IID generated data; target is its realized sample mean, not E[X]=0.
    return dict(average_variance=variance,mean_bias=bias,expected_mse=risk/n+(1-1/n)*bias*bias)
