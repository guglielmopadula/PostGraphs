
import numpy as np
import torch
'''
h=np.load("outputs.npy")
x=np.load("inputs.npy")
p=np.load("points.npy")

p=p.reshape(100,-1)
torch.manual_seed(0)
from vae import VAE
red=VAE(p[0].shape[0],500,1,0.5,True)
red.fit(p)    
par_red=red.compute_latent(p)
par_new,p_new=red.generate_new_dataset(100)
np.save("inputs_red.npy",par_new)
np.save("points_red.npy",p_new)
'''

'''
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import meshio


def compute(points_red):
    
    msh=meshio.read("reference.xdmf")
    points=points_red.reshape(-1,2)
    
    triangles=np.array(msh.cells_dict["triangle"])
    shape="triangle"
    degree = 1
    cell = ufl.Cell(shape, geometric_dimension=2)

    msh = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
    msh = mesh.create_mesh(MPI.COMM_WORLD, triangles, points, msh)
    tdim=msh.topology.dim
    fdim=tdim-1
    msh.topology.create_connectivity(fdim, tdim)
    V = fem.FunctionSpace(msh, ("Lagrange", 3))
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1,entities=boundary_facets)
    
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx + inner(u, v) * dx
    x = ufl.SpatialCoordinate(msh)
    f =(x[0]) ** 2 + (x[1]) ** 2    
    L = inner(f, v) * dx 
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return np.array(uh.vector)  
points_red=np.load("points_red.npy")
'''

'''
test=compute(points_red[0])
outputs_red=np.zeros((100,*test.shape))
for i in range(100):  
    outputs_red[i]=compute(points_red[i])

np.save("outputs_red.npy",outputs_red)
'''



h_red=np.load("outputs_red.npy")
x_red=np.load("inputs_red.npy")




import torch
import matplotlib.pyplot as plt
'''
np.random.seed(0)
torch.manual_seed(0)
n=[5,10,20,50,100]
loss_garom_red=np.zeros(5)
from garom import GAROM
for i in range(5):
    model=GAROM(h_red[0].shape[0],n[i],1,5)
    model.fit(x_red,h_red)    
    model.predict(x_red)
    loss_garom_red[i]=torch.linalg.norm(torch.tensor(h_red).float()-model.predict(x_red))/torch.linalg.norm(torch.tensor(h_red).float())

np.save("loss_garom_red.npy",loss_garom_red)
'''
from ezyrb import ANN
from ezyrb import POD
from ezyrb import ReducedOrderModel
from ezyrb import Database
from torch import nn

class customAct(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.relu=nn.ReLU()
        self.batchnorm=nn.BatchNorm1d(n)

    def forward(self,x):
        x=self.relu(x)
        x=self.batchnorm(x)
        return x

'''
np.random.seed(0)
torch.manual_seed(0)
loss_ann_red=np.zeros(5)
n=[5,10,20,50,100]
for i in range(5):
    ann=ANN([n[i],n[i]],customAct(n[i]),2000,lr=0.01,l2_regularization=0.00001)
    ann.fit(x_red,h_red)
    loss_ann_red[i]=np.linalg.norm(h_red-ann.predict(x_red))/np.linalg.norm(h_red)

np.save("loss_ann_red.npy",loss_ann_red)

'''
'''
np.random.seed(0)
torch.manual_seed(0)
loss_podann=np.zeros(5)
n=[5,10,20,50,100]

for i in range(5):
    db=Database(x_red,h_red)
    pod=POD('svd',rank=100)
    nen=ANN([n[i],n[i]],customAct(n[i]),3000,lr=0.01,l2_regularization=0.00001)
    rom=ReducedOrderModel(db,pod,nen)
    rom.fit()
    loss_podann[i]=np.linalg.norm(h_red-rom.predict(x_red))/np.linalg.norm(h_red)

np.save("loss_podann_red.npy",loss_podann)
'''
'''
loss_vae_red=np.zeros(5)
n=[5,10,20,50,100]
np.random.seed(0)
torch.manual_seed(0)
from vae import VAE
red=VAE(h_red[0].shape[0],100,10,0.005)
red.fit(h_red)    
z=red.compute_latent(h_red)

for i in range(5):
    ann=ANN([n[i],n[i]],customAct(n[i]),2000,lr=0.001)
    ann.fit(x_red,z)
    loss_vae_red[i]=np.linalg.norm(h_red-red.reconstruct(ann.predict(x_red)))/np.linalg.norm(h_red)
np.save("loss_vae_red.npy",loss_vae_red)

'''

loss_vae_red=np.load("loss_vae_red.npy")
loss_vae=np.load("loss_vae.npy")
loss_podann_red=np.load("loss_podann_red.npy")
loss_podann=np.load("loss_podann.npy")
loss_ann_red=np.load("loss_ann_red.npy")
loss_ann=np.load("loss_ann.npy")
loss_garom_red=np.load("loss_garom_red.npy")
loss_garom=np.load("loss_garom.npy")

n=[5,10,20,50,100]
fig,axis=plt.subplots(2,2)

axis[0,0].set_ylabel("relative error")
axis[0,0].plot(n,loss_ann,label="non reduced")
axis[0,0].plot(n,loss_ann_red,label="reduced")
axis[0,0].legend()
axis[0,0].title.set_text("FANN")
axis[0,0].set_xticks([])

axis[0,1].plot(n,loss_podann,label="non reduced")
axis[0,1].plot(n,loss_podann_red,label="reduced")
axis[0,1].legend()
axis[0,1].title.set_text("POD+FANN")
axis[0,1].set_xticks([])


axis[1,0].set_xlabel("size of the network")
axis[1,0].set_ylabel("relative error")
axis[1,0].plot(n,loss_vae,label="non reduced")
axis[1,0].plot(n,loss_vae_red,label="reduced")
axis[1,0].legend()
axis[1,0].title.set_text("VAE+FANN")

axis[1,1].set_xlabel("size of the network")
axis[1,1].plot(n,loss_garom,label="non reduced")
axis[1,1].plot(n,loss_garom_red,label="reduced")
axis[1,1].legend()
axis[1,1].title.set_text("GAROM")

fig.savefig("redvsnonred.png")
