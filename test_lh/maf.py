import jax
import jax.numpy as jnp
import evox as ex
from evox.operators.sampling import UniformSampling, LatinHypercubeSampling
from scipy.spatial.distance import pdist
import chex
from functools import partial
from jax import vmap
from matplotlib.path import Path
import numpy as np
class MaF(ex.Problem):
    """MAF"""

    def __init__(self, d=None, m=None, ref_num=1000,):
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 4
        else:
            self.d = d
        self._maf = None
        self.ref_num = ref_num

    def setup(self, key):
        return ex.State(key=key)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        chex.assert_type(X, float)
        chex.assert_shape(X, (None, self.n))
        return jax.jit(jax.vmap(self._maf))(X), state

    def pf(self, state: chex.PyTreeDef):
        f = 1 - UniformSampling(self.ref_num * self.m,self.m).random()[0]
        # f = LatinHypercubeSampling(self.ref_num * self.m, self.m).random(state.key)[0] / 2
        return f, state

class MaF1(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        n,d = jnp.shape(X)
        g = jnp.sum((X[:,m-1:]-0.5).__pow__(2),axis=1)
        f = jnp.repeat(1+g,m,axis=1) - jnp.repeat(1+g,m,axis=1)*jnp.fliplr(jnp.cumprod(jnp.concatenate((jnp.ones((n,1)),X[:,:m-1]),axis=1),axis=1))*jnp.concatenate((jnp.ones(n,1), 1-X[:,m-2::-1]),axis=1)
        return f, state 

    def pf(self, state: chex.PyTreeDef):
        f = 1 - UniformSampling(self.ref_num * self.m,self.m).random()[0];
        return f, state

class MaF2(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        m = self.m
        n,d = jnp.shape(X)
        g = jnp.zeros((n,m))
        for i in range(m):
            if i < m-1:
                temp1 = X[:,m+i*jnp.floor((d-m+1)/m):m+(i+1)*jnp.floor((d-m+1)/m)-1]/2 + 1/4
                g = g.at[:,i].set(jnp.sum((temp1-0.5)._pow__(2),axis=1))
            else:
                temp1 = X[:,m+(m-1)*jnp.floor((d-m+1)/m):m]/2 + 1/4
                g = g.at[:,i].set(jnp.sum((temp1-0.5)._pow__(2),axis=1))
            f1 = jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((n,1)),jnp.cos((X[:,:m]/2+1/4)*jnp.pi/2)]),axis=1))
            f2 = jnp.hstack([jnp.ones((n,1)),jnp.sin(((X[:,m-1::-1])/2+1/4)*jnp.pi/2)])
            f = (1+g)*f1*f2
        return f, state 

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        r = UniformSampling(n,self.m).random()[0];
        c = jnp.zeros((n, self.m-1))
        for i in range(n):
            for j in range(1,self.m):
                temp = r[i,j]/r[i,0]*jnp.prod(c[i,self.m-j+1:self.m-1])
                c = c.at[i,self.m-j].set(jnp.sqrt(1/(1+temp.__pow__(2))))
        if self.m > 5:
            c = c*(jnp.cos(jnp.pi/8)-jnp.cos(3*jnp.pi/8))+jnp.cos(3*jnp.pi/8)
        else:
            temp = jnp.any(jnp.logical_or(c<jnp.cos(3*jnp.pi/8),c>jnp.cos(jnp.pi/8)),axis=1)
            c = jnp.delete(c, temp.flatten() == 1, axis=0)
            
        n, _ = jnp.shape(c)
        f = jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((n,1)),c[:,:self.m-1]]),2)) * jnp.hstack([jnp.ones(n,1),jnp.sqrt(1-c[:, self.m-2::-1].__pow__(2))])
        return f, state
            
class MaF3(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n,d = jnp.shape(X)
        g = 100*(d-self.m+1+jnp.sum((X[:,self.m-1:]-0.5).__pow__(2)-jnp.cos(20*jnp.pi*(X[:,self.m-1:]-0.5)),axis=1));
        f1 = jnp.repeat(1+g,self.m,axis=1)*jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((n,1)),jnp.cos(X[:,:self.m-2]*jnp.pi/2)]),axis=1))*jnp.hstack([jnp.ones((n,1)),jnp.sin(X[:,self.m-2::-1]*jnp.pi/2)])
        f = jnp.hstack([f1[:,:self.m-2].__pow__(4), f1[:,self.m-1].__pow__(2)]);
        return f, state 

    def pf(self, state: chex.PyTreeDef):
        r = UniformSampling(self.ref_num * self.m,self.m).random()[0].__pow__(2)
        temp = jnp.sum(jnp.sqrt(r[:,:-1]),axis=1) + r[:,-1]
        f = r/jnp.stack([jnp.repeat(temp.__pow__(2),self.d-1,axis=1), temp])
        return f, state
    
class MaF4(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n,d = jnp.shape(X)
        g = 100*(d-self.m+1+jnp.sum((X[:,self.m-1:]-0.5).__pow__(2)-jnp.cos(20*jnp.pi*(X[:,self.m-1:]-0.5)),axis=1));
        f1 = jnp.repeat(1+g,self.m,axis=1) - jnp.repeat(1+g,self.m,axis=1)*jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((n,1)),jnp.cos(X[:,:self.m-2]*jnp.pi/2)]),axis=1))*jnp.hstack([jnp.ones((n,1)),jnp.sin(X[:,self.m-2::-1]*jnp.pi/2)])
        f = f1 * jnp.repeat(jnp.power(2, jnp.arange(1, self.m+1)),n,axis=0)
        return f, state 

    def pf(self, state: chex.PyTreeDef):
        r = UniformSampling(self.ref_num * self.m,self.m).random()[0]
        r1 = r / jnp.repeat(jnp.sqrt(jnp.sum(r.__pow__(2), axis=1)), self.m, axis=1)
        f = (1-r1)* jnp.repeat(jnp.power(2, jnp.arange(1, self.m+1)),self.ref_num * self.m,axis=0)
        return f, state
    
class MaF5(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n,d = jnp.shape(X)
        X = X.at[:,:self.m-1].set(X[:,:self.m-1].__pow__(100));
        g = jnp.sum((X[:,self.m-1:]-0.5).__pow__(2), axis=1);
        f1 = jnp.repeat(1+g,self.m,axis=1)*jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((n,1)),jnp.cos(X[:,:self.m-1]*jnp.pi/2)]),axis=1))*jnp.hstack([jnp.ones((n,1)),jnp.sin(X[:,self.m-2::-1]*jnp.pi/2)])
        f = f1*jnp.repeat(jnp.power(2, jnp.arange(self.m, 0,-1)),n,axis=0);
        return f, state 

        # R = R.*repmat(2.^(obj.M:-1:1),size(R,1),1);
    def pf(self, state: chex.PyTreeDef):
        r = UniformSampling(self.ref_num * self.m,self.m).random()[0]
        r1 = r / jnp.repeat(jnp.sqrt(jnp.sum(r.__pow__(2), axis=1)), self.m, axis=1)
        f = r1 * jnp.repeat(jnp.power(2, jnp.arange(1, self.m+1)), self.ref_num * self.m,axis=0)
        return f, state

class MaF6(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n,d = jnp.shape(X)
        i = 2
        g = jnp.sum((X[:,self.m-1:]-0.5).__pow__(2), axis=1);
        temp = jnp.repeat(g, self.m-i, axis=1)
        X = X.at[:,i-1:self.m-1].set((1+2*temp*X[:,i-1:self.m-1])/(2+2*temp))
        f = jnp.repeat(1+100*g,self.m,axis=1) * jnp.fliplr(jnp.cumprod(jnp.hstack([jnp.ones((n,1)), jnp.cos(X[:,:self.m-1]*jnp.pi/2)]),axis=1)) * jnp.hstack([jnp.ones((n,1)), jnp.sin(X[:,self.m-2::-1]*jnp.pi/2)])
        return f, state 

    def pf(self, state: chex.PyTreeDef):
        i = 2;
        n = self.ref_num * self.m
        r = UniformSampling(n,i).random()[0]
        r1 = r / jnp.repeat(jnp.sqrt(jnp.sum(r.__pow__(2),axis=1)),i,axis=1)
        r2 = jnp.hstack([jnp.repeat(r1[:,1], self.m-i, axis=1), r1])
        f = r2 / jnp.power(jnp.sqrt(2),jnp.repeat(jnp.maximum(self.m-i, jnp.arange(self.m-i,2-i,-1)), n,axis=0))
        return f, state
    
class MaF7(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n,d = jnp.shape(X)
        f = jnp.zeros((n,self.m))
        g = 1 + 9*jnp.mean(X[:,self.m-1:],axis=1)
        f = f.at[:,:self.m-2].set(X[:,:self.m-2])
        f = f.at[:,self.m-1].set((1+g)*(self.m-jnp.sum(X[:,:self.m-2]/(1+jnp.repeat(g,self.m-1,axis=1))*(1+jnp.sin(3*jnp.pi*X[:,:self.m-2:])),axis=1)))
        return f,state
    
    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        interval = jnp.array([0,0.251412,0.631627,0.859401])
        median = (interval[1]-interval[0])/(interval[3]-interval[2]+interval[1]-interval[0])
        X = UniformSampling(n,self.m-1).random()[0]
        X = jnp.where(X <= median, X * (interval[1]-interval[0])/median+interval[0], X)
        X = jnp.where(X > median, (X-median) * (interval[3]-interval[2])/(1-median)+interval[2], X)    
        f = jnp.hstack([X, 2*(self.m-jnp.sum(X/2*(1+jnp.sin(3*jnp.pi*X)),axis=1))])
        return f, state
    
class MaF8(MaF):
    """
    the dimention only is 2.
    """
    def __init__(self, d=None, m=None, ref_num=1000):
        d = 2
        super().__init__(d, m, ref_num)
        self.points = self._getPoints()
    
    
    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        f = self._eucl_dis(X, self.points)
        return f, state


    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # [X, Y] = ndgrid(linspace(-1, 1, ceil(sqrt(N))));
        temp = jnp.linspace(-1,1,num=jnp.ceil(jnp.sqrt(n)).astype(jnp.int32))
        x, y = jnp.meshgrid(temp, temp)
        x = x.ravel(order="F")
        y = y.ravel(order="F")
        # using jnp as np, this may make some mistakes, but in my test, there is no warning
        poly_path = Path(self.points)
        _points = jnp.column_stack((x, y))
        ND = poly_path.contains_points(_points)

        f = self._eucl_dis([x[ND], y[ND]], self.points)
        return f, state

    def _getPoints(self):
        thera, rho = self._cart2pol(0, 1)
        temp = jnp.arange(1, self.m + 1).reshape((-1, 1))
        x, y = self._pol2cart(thera - temp * 2 * jnp.pi / self.m, rho)
        return jnp.vstack([x, y])

    def _cart2pol(self, x, y):
        rho = jnp.sqrt(jnp.power(x, 2) + jnp.power(y, 2))
        theta = jnp.arctan2(y, x)
        return theta, rho

    def _pol2cart(self, theta, rho):
        x = rho * jnp.cos(theta)
        y = rho * jnp.sin(theta)
        return (x, y)
    def _eucl_dis(self,x,y):
        a_squared = jnp.sum(jnp.square(x), axis=1)
        b_squared = jnp.sum(jnp.square(y), axis=1)
        ab = jnp.dot(x, y.T)
        distance = jnp.sqrt(jnp.add(jnp.add(-2 * ab, a_squared[:, jnp.newaxis]), b_squared))
        return distance



class MaF9(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        super().__init__(d, m, ref_num)
        # Generate vertexes
        self.points = self._getPoints()

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        m, d1 = jnp.shape(self.points)
        f = jnp.zeros((n,m))
        for i in range(m):
            f = f.at[:,i].set(self._Point2Line(X, self.points[jnp.mod(jnp.arange(i, i+2),m)+1,:]))
        return f, state

    def pf(self, state: chex.PyTreeDef):
        n = self.ref_num * self.m
        # [X, Y] = ndgrid(linspace(-1, 1, ceil(sqrt(N))));
        temp = jnp.linspace(-1, 1, num=jnp.ceil(jnp.sqrt(n)).astype(jnp.int32))
        x, y = jnp.meshgrid(temp, temp)
        x = x.ravel(order="F")
        y = y.ravel(order="F")
        # using jnp as np, this may make some mistakes, but in my test, there is no warning
        poly_path = Path(self.points)
        _points = jnp.column_stack((x, y))
        ND = poly_path.contains_points(_points)

        f = self.evaluate(state, jnp.vstack((x[ND], y[ND])))
        return f, state

    def _getPoints(self):
        thera, rho = self._cart2pol(0, 1)
        temp = jnp.arange(1, self.m + 1).reshape((-1, 1))
        x, y = self._pol2cart(thera - temp * 2 * jnp.pi / self.m, rho)
        return jnp.vstack([x, y])

    def _cart2pol(self, x, y):
        rho = jnp.sqrt(jnp.power(x, 2) + jnp.power(y, 2))
        theta = jnp.arctan2(y, x)
        return theta, rho

    def _pol2cart(self, theta, rho):
        x = rho * jnp.cos(theta)
        y = rho * jnp.sin(theta)
        return (x, y)

    def _Point2Line(self, f, line):
        distance = jnp.abs((line[0,0] - f[:,0]) * (line[1,1] - f[:,1]) - (line[1,0] - f[:,0]) * (line[0,1] - f[:,1]))  / jnp.sqrt((line[0,0]-line[1,0])**2 + (line[0,1]-line[1,1])**2)
        return distance

class MaF10(MaF):
    def __init__(self, d=None, m=None, ref_num=1000):
        if m is None:
            self.m = 3
        else:
            self.m = m
        if d is None:
            self.d = self.m + 9
        else:
            self.d = d
        super().__init__(d, m, ref_num)

    def evaluate(self, state: chex.PyTreeDef, X: chex.Array):
        n, d = jnp.shape(X)
        M = self.m
        K = M - 1
        L = d - K
        D = 1
        S = jnp.arange(2, 2*M+1, 2)
        A = jnp.ones((M-1))

        z01 = X / jnp.repeat(jnp.arange(2,d*2+1,2), n, axis=0)
        t1 = jnp.zeros((n, K+L))
        t1 = t1.at[:,:K].set(z01[:,:K])
        t1 = t1.at[:,K:].set(self._s_linear(z01[:,K:],0.35))

        t2 = jnp.zeros((n,K+L))
        t2 = t2.at[:,:K].set(t1[:,:K])
        t2 = t2.at[:,K:].set(self._b_flat(t1[:,K:], 0.8,0.75,0.85))

        

    def _s_linear(self,y, A):
        output = jnp.abs(y-A).jnp.abs((A-y).astype(jnp.float32) + A)
        return output

    def _b_flat(self,y, A, B, C):
        output = A + jnp.minimum(0, jnp.floor(y-B))*A*(B-y)/B - jnp.minimum(0, jnp.floor(C - y))*(1-A)*(y-C)/(1-C)
        output = jnp.round(output*1e4)/1e4
        return output




        
    
    
