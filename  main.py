import numpy as np

from tqdm import trange
from torch import nn
import torch
import matplotlib.pyplot as plt

'''
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

def compute(mu):
    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (mu, mu)), n=(100, 100),
                            cell_type=mesh.CellType.triangle,)
    V = fem.FunctionSpace(msh, ("Lagrange", 1))
    facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], mu)),np.logical_or(np.isclose(x[1], 0.0),
                                                                      np.isclose(x[1], mu))))
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1,entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx + mu*inner(u, v) * dx
    x = ufl.SpatialCoordinate(msh)
    f =(x[0] - mu/2) ** 2 + (x[1] - mu/2) ** 2    
    L = inner(f, v) * dx 
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return np.array(uh.vector)  

test=compute(1.0)
h=np.zeros((101,*test.shape))
for i in trange(101):
  h[i]=compute(1.0+4.0*i/100)  
np.save("fem_sol.npy",h)

'''

class customAct(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.relu=nn.ReLU()
        self.batchnorm=nn.BatchNorm1d(n)

    def forward(self,x):
        x=self.relu(x)
        x=self.batchnorm(x)
        return x

h=np.load("fem_sol.npy")
x=1.0+4.0*np.arange(101)/100
x=x.reshape(-1,1)
from ezyrb import ANN
from ezyrb import POD
from ezyrb import ReducedOrderModel
from ezyrb import Database


'''
loss_ann=np.zeros(5)
n=[5,10,20,50,100]
for i in range(5):
    ann=ANN([n[i],n[i]],customAct(n[i]),2000)
    ann.fit(x,h)
    loss_ann[i]=np.linalg.norm(h-ann.predict(x))/np.linalg.norm(h)


plt.plot(n,loss_ann)
plt.savefig("ann.pdf")

np.save("loss_ann.npy",loss_ann)
'''

'''
n=[5,10,20,50,100]
loss_garom=np.zeros(5)
from garom import GAROM
for i in range(5):
    model=GAROM(h[0].shape[0],n[i],1,5)
    model.fit(x,h)    
    model.predict(x)
    loss_garom[i]=torch.linalg.norm(torch.tensor(h).float()-model.predict(x))/torch.linalg.norm(torch.tensor(h).float())

np.save("loss_garom.npy",loss_garom)
plt.plot(n,loss_garom)
plt.savefig("garom.pdf")

'''

'''
loss_podann=np.zeros(5)
n=[5,10,20,50,100]


for i in range(5):
    db=Database(x,h)
    pod=POD('svd')
    nen=ANN([n[i],n[i]],customAct(n[i]),2000)
    rom=ReducedOrderModel(db,pod,nen)
    rom.fit()
    loss_podann[i]=np.linalg.norm(h-rom.predict(x))/np.linalg.norm(h)


np.save("loss_podann.npy",loss_podann)

plt.plot(n,loss_podann)
plt.savefig("podann.pdf")
'''

'''
loss_vae=np.zeros(5)
n=[5,10,20,50,100]
from vae import VAE
for i in range(5):
    red=VAE(h[0].shape[0],n[i],5)
    red.fit(h)    
    z=red.compute_latent(h)
    ann=ANN([n[i],n[i]],customAct(n[i]),2000)
    ann.fit(x,z)
    loss_vae[i]=np.linalg.norm(h-red.reconstruct(ann.predict(x)))/np.linalg.norm(h)
    
np.save("loss_vae.npy",loss_vae)
 
plt.plot(n,loss_vae)
plt.savefig("vaeann.pdf")
'''
n=[5,10,20,50,100]

loss_vae=np.load("loss_vae.npy")
loss_garom=np.load("loss_garom.npy")
loss_ann=np.load("loss_ann.npy")
loss_podann=np.load("loss_podann.npy")

plt.plot(n,loss_vae,label="VAE+FANN")
plt.plot(n,loss_garom,label="GAROM")
plt.plot(n,loss_ann,label="FANN")
plt.plot(n,loss_podann,label="POD+FANN")
plt.legend()
plt.savefig("all.png")
'''

plt.plot(n,loss_ann)
plt.xlabel("size of the network")
plt.ylabel("relative error")
plt.suptitle("FANN")
plt.savefig("ann.png")

'''
'''
plt.plot(n,loss_vae)
plt.xlabel("size of the network")
plt.suptitle("VAE+FANN")
plt.ylabel("relative error")
plt.savefig("vae.png")
'''

'''
plt.plot(n,loss_garom)
plt.xlabel("size of the network")
plt.suptitle("Conditional GAN (aka GAROM)")
plt.ylabel("relative error")
plt.savefig("garom.png")
'''

'''
plt.plot(n,loss_podann)
plt.xlabel("size of the network")
plt.suptitle("POD+FANN")
plt.ylabel("relative error")
plt.savefig("podann.png")
'''