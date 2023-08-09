import numpy as np

from tqdm import trange
from torch import nn
import torch
import matplotlib.pyplot as plt
import meshio

'''

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

msh = mesh.create_unit_square(comm=MPI.COMM_WORLD,
        nx=10,ny=10,
        cell_type=mesh.CellType.triangle,)

with io.XDMFFile(msh.comm, "reference.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)


def compute(mu,gamma):
    msh=meshio.read("reference.xdmf")
    points=np.array(msh.points)
    points=points-0.5
    points[:,0]=points[:,0]*mu
    points[:,1]=points[:,1]*gamma
    triangles=np.array(msh.cells_dict["triangle"])
    shape="triangle"
    degree = 1
    cell = ufl.Cell(shape, geometric_dimension=2)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
    msh = mesh.create_mesh(MPI.COMM_WORLD, triangles, points, domain)
    V = fem.FunctionSpace(msh, ("Lagrange", 3))
    facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], -mu/2),
                                                                      np.isclose(x[0], mu/2)),np.logical_or(np.isclose(x[1], -gamma/2),
                                                                      np.isclose(x[1], gamma/2))))
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1,entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx + inner(u, v) * dx
    x = ufl.SpatialCoordinate(msh)
    f =(x[0]) ** 2 + (x[1]) ** 2    
    L = inner(f, v) * dx 
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return np.array(uh.vector),points  



test,points=compute(1.0,1.0)
h=np.zeros((100,*test.shape))
x=np.zeros((100,2))
mu=np.linspace(0.9,1.1,10)
gamma=np.linspace(0.9,1.1,10)
tot_points=np.zeros((100,*points.shape))

for i in trange(100):
    h[i],points=compute(mu[i%10],gamma[i//10])  
    x[i]=np.array([mu[i%10],gamma[i//10]])
    tot_points[i]=points
np.save("outputs.npy",h)
np.save("inputs.npy",x)
np.save("points.npy",tot_points)


'''



from ezyrb import ANN
from ezyrb import POD
from ezyrb import ReducedOrderModel
from ezyrb import Database


class customAct(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.relu=nn.ReLU()
        self.batchnorm=nn.BatchNorm1d(n)

    def forward(self,x):
        x=self.relu(x)
        x=self.batchnorm(x)
        return x

h=np.load("outputs.npy")
x=np.load("inputs.npy")

'''
np.random.seed(0)
torch.manual_seed(0)
loss_ann=np.zeros(5)
n=[5,10,20,50,100]
for i in range(5):
    ann=ANN([n[i],n[i]],customAct(n[i]),2000,lr=0.01,l2_regularization=0.00001) #1000,0.001
    ann.fit(x,h)
    loss_ann[i]=np.linalg.norm(h-ann.predict(x))/np.linalg.norm(h)


np.save("loss_ann.npy",loss_ann)
plt.plot(n,loss_ann)
plt.xlabel("size of the network")
plt.suptitle("FANN")
plt.ylabel("relative error")
plt.savefig("ann.png")
'''




'''

np.random.seed(0)
torch.manual_seed(0)
n=[5,10,20,50,100]
loss_garom=np.zeros(5)
from garom import GAROM
for i in range(5):
    model=GAROM(h[0].shape[0],n[i],2,5)
    model.fit(x,h)    
    model.predict(x)
    loss_garom[i]=torch.linalg.norm(torch.tensor(h).float()-model.predict(x))/torch.linalg.norm(torch.tensor(h).float())

np.save("loss_garom.npy",loss_garom)
plt.plot(n,loss_garom)
plt.xlabel("size of the network")
plt.suptitle("Conditional GAN (aka GAROM)")
plt.ylabel("relative error")
plt.savefig("garom.png")
'''


'''
np.random.seed(0)
torch.manual_seed(0)
loss_podann=np.zeros(5)
n=[5,10,20,50,100]

for i in range(5):
    db=Database(x,h)
    pod=POD('svd',rank=100)
    nen=ANN([n[i],n[i]],customAct(n[i]),3000,lr=0.01,l2_regularization=0.00001)
    rom=ReducedOrderModel(db,pod,nen)
    rom.fit()
    loss_podann[i]=np.linalg.norm(h-rom.predict(x))/np.linalg.norm(h)
print(loss_podann[4])

np.save("loss_podann.npy",loss_podann)

n=[5,10,20,50,100]
loss_podann=np.load("loss_podann.npy")
plt.plot(n,loss_podann)
plt.xlabel("size of the network")
plt.suptitle("POD+FANN")
plt.ylabel("relative error")
plt.savefig("podann.png")
'''



'''
loss_vae=np.zeros(5)
n=[5,10,20,50,100]
np.random.seed(0)
torch.manual_seed(0)
from vae import VAE
red=VAE(h[0].shape[0],100,10,0.005)
red.fit(h)    
z=red.compute_latent(h)

for i in range(5):
    ann=ANN([n[i],n[i]],customAct(n[i]),2000,lr=0.001)
    ann.fit(x,z)
    loss_vae[i]=np.linalg.norm(h-red.reconstruct(ann.predict(x)))/np.linalg.norm(h)
np.save("loss_vae.npy",loss_vae)
n=[5,10,20,50,100]
loss_vae=np.load("loss_vae.npy")
plt.plot(n,loss_vae)
plt.xlabel("size of the network")
plt.suptitle("VAE+FANN")
plt.ylabel("relative error")
plt.savefig("vae.png")
'''


'''
hred=np.load("outputsgen.npy")
xred=np.load("inputsgen.npy")
n=[5,10,20,50,100]
loss_garom_red=np.zeros(5)
from garom import GAROM
for i in range(5):
    model=GAROM(h[0].shape[0],n[i],1,5)
    model.fit(xred,hred)    
    model.predict(xred)
    loss_garom_red[i]=torch.linalg.norm(torch.tensor(hred).float()-model.predict(xred))/torch.linalg.norm(torch.tensor(hred).float())

np.save("loss_garom_red.npy",loss_garom_red)
plt.plot(n,loss_garom_red)
plt.savefig("garom_red.png")
'''



n=[5,10,20,50,100]
loss_vae=np.load("loss_vae.npy")
loss_garom=np.load("loss_garom.npy")
loss_ann=np.load("loss_ann.npy")
loss_podann=np.load("loss_podann.npy")
plt.plot(n,loss_vae,label="VAE+FANN")
plt.xlabel("size of the network")
plt.ylabel("relative error")
plt.plot(n,loss_garom,label="GAROM")
plt.plot(n,loss_ann,label="FANN")
plt.plot(n,loss_podann,label="POD+FANN")
plt.legend()
plt.savefig("all.png")


'''
n=[5,10,20,50,100]
loss_ann=np.load("loss_ann.npy")
plt.plot(n,loss_ann)
plt.xlabel("size of the network")
plt.ylabel("relative error")
plt.suptitle("FANN")
plt.savefig("ann.png")
'''

'''
n=[5,10,20,50,100]
loss_vae=np.load("loss_vae.npy")
plt.plot(n,loss_vae)
plt.xlabel("size of the network")
plt.suptitle("VAE+FANN")
plt.ylabel("relative error")
plt.savefig("vae.png")
'''

'''
n=[5,10,20,50,100]
loss_garom=np.load("loss_garom.npy")
plt.plot(n,loss_garom)
plt.xlabel("size of the network")
plt.ylabel("relative error")
plt.suptitle("GAROM")
plt.savefig("ann.png")
'''

'''
n=[5,10,20,50,100]
loss_podann=np.load("loss_podann.npy")
plt.plot(n,loss_podann)
plt.xlabel("size of the network")
plt.suptitle("POD+FANN")
plt.ylabel("relative error")
plt.savefig("podann.png")
'''

