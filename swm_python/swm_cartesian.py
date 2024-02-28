import numpy as np
import argparse
import matplotlib.pyplot as plt
#import cupy
#import gt4py

# Initialize model parameters
M = 64 # args.M
N = 64 # args.N
M_LEN = M + 1
N_LEN = N + 1
L_OUT = True # args.L_OUT
VIS = False
VIS_DT=10
ITMAX = 4000
dt = 90.
tdt = dt
dx = 100000.
dy = 100000.
fsdx = 4. / (dx)
fsdy = 4. / (dy)
a = 1000000.
alpha = 0.001
el = N * dx
pi = 4. * np.arctan(1.)
tpi = 2. * pi
d_i = tpi / M
d_j = tpi / N
pcf = (pi * pi * a * a) / (el * el)
SIZE = M_LEN * N_LEN

# Model Variables
u = np.zeros((M_LEN, N_LEN, 1))
v = np.zeros((M_LEN, N_LEN, 1))
p = np.zeros((M_LEN, N_LEN, 1))
unew = np.zeros((M_LEN, N_LEN, 1))
vnew = np.zeros((M_LEN, N_LEN, 1))
pnew = np.zeros((M_LEN, N_LEN, 1))
uold = np.zeros((M_LEN, N_LEN, 1))
vold = np.zeros((M_LEN, N_LEN, 1))
pold = np.zeros((M_LEN, N_LEN, 1))
uvis = np.zeros((M_LEN, N_LEN, 1))
vvis = np.zeros((M_LEN, N_LEN, 1))
pvis = np.zeros((M_LEN, N_LEN, 1))
cu = np.zeros((M_LEN, N_LEN, 1))
cv = np.zeros((M_LEN, N_LEN, 1))
z = np.zeros((M_LEN, N_LEN, 1))
h = np.zeros((M_LEN, N_LEN, 1))
psi = np.zeros((M_LEN, N_LEN, 1))

from IPython.display import clear_output
from matplotlib import pyplot as plt
    

def live_plot3(fu, fv, fp, title=''):
    clear_output(wait=True)
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

    pos1 = ax1.imshow(fp, cmap='Blues', vmin=49999, vmax=50001,interpolation='none')
    ax1.set_title('p')
    pos2 = ax2.imshow(fu, cmap='Reds', vmin=-1, vmax=1,interpolation='none')
    ax2.set_title('u')
    pos3 = ax3.imshow(fv, cmap='Greens',vmin=-1, vmax=1,interpolation='none')
    ax3.set_title('v')

    fig.suptitle(title)
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.show()

# Initial values of the stream function and p
for i in range(M + 1):
    for j in range(N + 1):
        psi[i, j, 0] = a * np.sin((i + .5) * d_i) * np.sin((j + .5) * d_j)
        p[i, j, 0] = pcf * (np.cos(2. * (i) * d_i) + np.cos(2. * (j) * d_j)) + 50000.
            
# Calculate initial u and v
    for i in range(M):
        for j in range(N):
            u[i+1,j, 0] = -(psi[i+1,j+1,0] - psi[i+1,j,0]) / dy
            v[i,j+1, 0] = (psi[i+1,j+1,0] - psi[i,j+1,0]) / dx
            

if VIS==True:
    live_plot3(u,v,p, "init")
    print(p.max())
    print(p.min())
    print(u.max())
    print(u.min())
    print(v.max())
    print(v.min())

# Periodic Boundary conditions

u[0, :,0] = u[M, :,0]
v[M, 1:,0] = v[0, 1:,0]
u[1:, N,0] = u[1:, 0,0]
v[:, 0,0] = v[:, N,0]

u[0, N,0] = u[M, 0,0]
v[M, 0,0] = v[0, N,0]


if VIS==True:
    live_plot3(u,v,p, "Periodic Bounday Conditions")
    
# Save initial conditions
uold = np.copy(u[...])
vold = np.copy(v[...])
pold = np.copy(p[...])


# In[6]:


# Print initial conditions
if L_OUT:
    print(" Number of points in the x direction: ", M)
    print(" Number of points in the y direction: ", N)
    print(" grid spacing in the x direction: ", dx)
    print(" grid spacing in the y direction: ", dy)
    print(" time step: ", dt)
    print(" time filter coefficient: ", alpha)
    print(" Initial p:\n", p[:,:,0].diagonal()[:-1])
    print(" Initial u:\n", u[:,:,0].diagonal()[:-1])
    print(" Initial v:\n", v[:,:,0].diagonal()[:-1])
        
import numpy as np
import gt4py.next as gtx
import gt4py.cartesian.gtscript as gtscript

nx = M
ny = N
nz = 1
dtype = np.float64
gt4py_type = "cartesian"
#gt4py_type = "next"
allocator = gtx.itir_python

I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K", kind = gtx.DimensionKind.VERTICAL)

domain = gtx.domain({I:nx+1, J:ny+1, K:nz})

h_gt = gtx.as_field(domain,h,allocator=allocator)
p_gt = gtx.as_field(domain,p,allocator=allocator)
u_gt = gtx.as_field(domain,u,allocator=allocator)
v_gt = gtx.as_field(domain,v,allocator=allocator)
z_gt = gtx.as_field(domain,z,allocator=allocator)
cu_gt = gtx.as_field(domain,cu,allocator=allocator)
cv_gt = gtx.as_field(domain,cv,allocator=allocator)

cartesian_backend = "numpy"
next_backend = gtx.itir_python

if gt4py_type == "cartesian":
    time = 0.0
    # Main time loop
    for ncycle in range(ITMAX):
        if((ncycle%100==0) & (VIS==False)):
            print("cycle number ", ncycle)
        # Calculate cu, cv, z, and h
        #for i in range(M):
        #    for j in range(N):
        #        h[i, j,0] = p[i, j,0] + 0.25 * (u[i + 1, j,0] * u[i + 1, j,0] + u[i, j,0] * u[i, j,0] +
        #                                v[i, j + 1,0] * v[i, j + 1,0] + v[i, j,0] * v[i, j,0])
        # i --> 0,M
        #j --> 0,N
        # at nx+1 its the boundary region, mask one region and 
        @gtscript.stencil(backend=cartesian_backend)
        def calc_h(
            p: gtscript.Field[dtype],
            u: gtscript.Field[dtype],
            v: gtscript.Field[dtype],
            h: gtscript.Field[dtype]
        ):
            with computation(PARALLEL), interval(...):
                h = p + 0.25 * u[1,0,0] * u[1,0,0] + u * u + v[0,1,0] * v[0,1,0] + v * v
    
        calc_h(p=p_gt, u=u_gt, v=v_gt, h=h_gt, origin=(0,0,0), domain=(nx,ny,nz))
    
        h = h_gt.asnumpy()

        #nx = M
        #ny = N
        #nz = 1
        # i --> 1,M+1  (1,1,0) (nx,ny,nz)
        #j --> 1,N+1
        @gtscript.stencil(backend=cartesian_backend)
        def calc_z(
            fsdx: float,
            fsdy: float,
            u: gtscript.Field[dtype],
            v: gtscript.Field[dtype],
            p: gtscript.Field[dtype],
            z: gtscript.Field[dtype]
        ):
            with computation(PARALLEL), interval(...):
                z = (fsdx * (v - v[-1,0,0]) - fsdy * (u - u[0,-1,0])) / (p[-1,-1,0] + p[0,-1,0] + p + p[-1,0,0])

        calc_z(fsdx=fsdx, fsdy=fsdy, u=u_gt, v=v_gt, p=p_gt, z=z_gt, origin=(1,1,0), domain=(nx,ny,nz)) # domain(nx+1,ny+1,nz) gives error why?
        z = z_gt.asnumpy()

        @gtscript.stencil(backend=cartesian_backend)
        def calc_cu(
            u: gtscript.Field[dtype],
            p: gtscript.Field[dtype],
            cu: gtscript.Field[dtype]
        ):
            with computation(PARALLEL), interval(...):
                cu = .5 * (p + p) * u

        #for i in range(1,M+1):
        #    for j in range(N):
        #        cu2[i, j,0] = .5 * (p[i, j,0] + p[i, j,0]) * u[i, j,0]
        calc_cu(u=u_gt, p=p_gt, cu=cu_gt, origin=(1,0,0), domain=(nx,ny+1,nz)) # domain(nx+1,ny+1,nz) gives error why? try removing ny+1
        cu = cu_gt.asnumpy()

        @gtscript.stencil(backend=cartesian_backend)
        def calc_cv(
            v: gtscript.Field[dtype],
            p: gtscript.Field[dtype],
            cv: gtscript.Field[dtype]
        ):
            with computation(PARALLEL), interval(...):
                cv = .5 * (p + p) * v

        calc_cv(v=v_gt, p=p_gt, cv=cv_gt, origin=(0,1,0), domain=(nx+1,ny,nz)) # domain(nx+1,ny+1,nz) gives error why?
        cv = cv_gt.asnumpy()
    
        #for i in range(M):
        #    for j in range(N):
        #        #cu[i + 1, j,0] = .5 * (p[i + 1, j,0] + p[i, j,0]) * u[i + 1, j,0]
        #        cv[i, j + 1,0] = .5 * (p[i, j + 1,0] + p[i, j,0]) * v[i, j + 1,0]
        #        #z[i + 1, j + 1,0] = (fsdx * (v[i + 1, j + 1,0] - v[i, j + 1,0]) -
        #        #                fsdy * (u[i + 1, j + 1,0] - u[i+1, j,0] )
        #        #                ) / (p[i, j,0] + p[i + 1, j,0] + p[i + 1, j + 1,0] + p[i, j + 1,0])
    
            # # Periodic Boundary conditions
        #try region
        cu[0, :,0] = cu[M, :,0]
        h[M, :,0] = h[0, :,0]
        cv[M, 1:,0] = cv[0, 1:,0]
        z[0, 1:,0] = z[M, 1:,0]
        
        cv[:, 0,0] = cv[:, N,0]
        h[:, N,0] = h[:, 0,0]
        cu[1:, N,0] = cu[1:, 0,0]
        z[1:, N,0] = z[1:, 0,0]
            
        cu[0, N,0] = cu[M, 0,0]
        cv[M, 0,0] = cv[0, N,0]
        z[0, 0,0] = z[M, N,0]
        h[M, N,0] = h[0, 0,0]
            
        # Calclulate new values of u,v, and p
        tdts8 = tdt / 8.
        tdtsdx = tdt / dx
        tdtsdy = tdt / dy
        #print(tdts8, tdtsdx, tdtsdy)
    
    
        for i in range(M):
            for j in range(N):
                unew[i+1,j,0] = uold[i+1,j,0] + tdts8 * (z[i+1,j+1,0] + z[i+1,j,0]) * (cv[i+1,j+1,0] + cv[i+1,j,0] + cv[i,j+1,0] + cv[i,j,0]) - tdtsdx * (h[i+1,j,0] - h[i,j,0])
                vnew[i,j+1,0] = vold[i,j+1,0] - tdts8 * (z[i+1,j+1,0] + z[i,j+1,0]) * (cu[i+1,j+1,0] + cu[i+1,j,0] + cu[i,j+1,0] + cu[i,j,0]) - tdtsdy * (h[i,j+1,0] - h[i,j,0])
                pnew[i,j,0] = pold[i,j,0] - tdtsdx * (cu[i+1,j,0] - cu[i,j,0]) - tdtsdy * (cv[i,j+1,0] - cv[i,j,0])
                    
        
        # Periodic Boundary conditions
        unew[0, :,0] = unew[M, :,0]
        pnew[M, :,0] = pnew[0, :,0]
        vnew[M, 1:,0] = vnew[0, 1:,0]
        unew[1:, N,0] = unew[1:, 0,0]
        vnew[:, 0,0] = vnew[:, N,0]
        pnew[:, N,0] = pnew[:, 0,0]
        
        unew[0, N,0] = unew[M, 0,0]
        vnew[M, 0,0] = vnew[0, N,0]
        pnew[M, N,0] = pnew[0, 0,0]
        
        time = time + dt
    
        if(ncycle > 0):
            for i in range(M_LEN):
                for j in range(N_LEN):
                    uoldtemp=uold[i,j,0]
                    voldtemp=vold[i,j,0]
                    poldtemp=pold[i,j,0] 
                    uold[i,j,0] = u[i,j,0] + alpha * (unew[i,j,0] - 2. * u[i,j,0] + uoldtemp)
                    vold[i,j,0] = v[i,j,0] + alpha * (vnew[i,j,0] - 2. * v[i,j,0] + voldtemp)
                    pold[i,j,0] = p[i,j,0] + alpha * (pnew[i,j,0] - 2. * p[i,j,0] + poldtemp)
    
            for i in range(M_LEN):
                    for j in range(N_LEN):
                        u[i,j,0] = unew[i,j,0]
                        v[i,j,0] = vnew[i,j,0]
                        p[i,j,0] = pnew[i,j,0]
    
        else:
            tdt = tdt+tdt
    
            uold = np.copy(u[...])
            vold = np.copy(v[...])
            pold = np.copy(p[...])
            u = np.copy(unew[...])
            v = np.copy(vnew[...])
            p = np.copy(pnew[...])
    
        if((VIS == True) & (ncycle%VIS_DT==0)):
            live_plot3(u, v, p, "ncycle: " + str(ncycle))
            
    # Print initial conditions
    if L_OUT:
           print("cycle number ", ITMAX)
           print(" diagonal elements of p:\n", pnew[:,:,0].diagonal()[:-1])
           print(" diagonal elements of u:\n", unew[:,:,0].diagonal()[:-1])
           print(" diagonal elements of v:\n", vnew[:,:,0].diagonal()[:-1])


# gt4py NEXT part!!!!!!!!!!!!!!!!!!!!


if gt4py_type == "next":
    time = 0.0 
    # Main time loop
    for ncycle in range(ITMAX):
        if((ncycle%100==0) & (VIS==False)):
            print("cycle number ", ncycle)
        # Calculate cu, cv, z, and h
        #for i in range(M):
        #    for j in range(N):
        #        h[i, j,0] = p[i, j,0] + 0.25 * (u[i + 1, j,0] * u[i + 1, j,0] + u[i, j,0] * u[i, j,0] +
        #                                v[i, j + 1,0] * v[i, j + 1,0] + v[i, j,0] * v[i, j,0])
        Ioff = gtx.FieldOffset("I", source=I, target=(I,))
        Joff = gtx.FieldOffset("J", source=J, target=(J,))

        @gtx.field_operator
        def cal_c():
            return p + 0.25 * u[1,0,0] * u[1,0,0] + u * u + v[0,1,0] * v[0,1,0] + v * v
        
        @gtx.program(backend=next_backend)
        def calc_h_program(
            p: gtscript.Field[[I,J,K],dtype],
            u: gtscript.Field[[I,J,K],dtype],
            v: gtscript.Field[[I,J,K],dtype],
            h: gtscript.Field[[I,J,K],dtype]
        ):
            with computation(PARALLEL), interval(...):
                h = p + 0.25 * u[1,0,0] * u[1,0,0] + u * u + v[0,1,0] * v[0,1,0] + v * v
    
        calc_h(p=p_gt, u=u_gt, v=v_gt, h=h_gt, origin=(0,0,0), domain=(nx,ny,nz))
    
        h = h_gt.asnumpy()
    
        for i in range(M):
            for j in range(N):
                cu[i + 1, j,0] = .5 * (p[i + 1, j,0] + p[i, j,0]) * u[i + 1, j,0]
                cv[i, j + 1,0] = .5 * (p[i, j + 1,0] + p[i, j,0]) * v[i, j + 1,0]
                z[i + 1, j + 1,0] = (fsdx * (v[i + 1, j + 1,0] - v[i, j + 1,0]) -
                                fsdy * (u[i + 1, j + 1,0] - u[i+1, j,0] )
                                ) / (p[i, j,0] + p[i + 1, j,0] + p[i + 1, j + 1,0] + p[i, j + 1,0])
    
            # # Periodic Boundary conditions
        cu[0, :,0] = cu[M, :,0]
        h[M, :,0] = h[0, :,0]
        cv[M, 1:,0] = cv[0, 1:,0]
        z[0, 1:,0] = z[M, 1:,0]
        
        cv[:, 0,0] = cv[:, N,0]
        h[:, N,0] = h[:, 0,0]
        cu[1:, N,0] = cu[1:, 0,0]
        z[1:, N,0] = z[1:, 0,0]
            
        cu[0, N,0] = cu[M, 0,0]
        cv[M, 0,0] = cv[0, N,0]
        z[0, 0,0] = z[M, N,0]
        h[M, N,0] = h[0, 0,0]
            
        # Calclulate new values of u,v, and p
        tdts8 = tdt / 8.
        tdtsdx = tdt / dx
        tdtsdy = tdt / dy
        #print(tdts8, tdtsdx, tdtsdy)
    
    
        for i in range(M):
            for j in range(N):
                unew[i+1,j,0] = uold[i+1,j,0] + tdts8 * (z[i+1,j+1,0] + z[i+1,j,0]) * (cv[i+1,j+1,0] + cv[i+1,j,0] + cv[i,j+1,0] + cv[i,j,0]) - tdtsdx * (h[i+1,j,0] - h[i,j,0])
                vnew[i,j+1,0] = vold[i,j+1,0] - tdts8 * (z[i+1,j+1,0] + z[i,j+1,0]) * (cu[i+1,j+1,0] + cu[i+1,j,0] + cu[i,j+1,0] + cu[i,j,0]) - tdtsdy * (h[i,j+1,0] - h[i,j,0])
                pnew[i,j,0] = pold[i,j,0] - tdtsdx * (cu[i+1,j,0] - cu[i,j,0]) - tdtsdy * (cv[i,j+1,0] - cv[i,j,0])
                    
        
        # Periodic Boundary conditions
        unew[0, :,0] = unew[M, :,0]
        pnew[M, :,0] = pnew[0, :,0]
        vnew[M, 1:,0] = vnew[0, 1:,0]
        unew[1:, N,0] = unew[1:, 0,0]
        vnew[:, 0,0] = vnew[:, N,0]
        pnew[:, N,0] = pnew[:, 0,0]
        
        unew[0, N,0] = unew[M, 0,0]
        vnew[M, 0,0] = vnew[0, N,0]
        pnew[M, N,0] = pnew[0, 0,0]
        
        time = time + dt
    
        if(ncycle > 0):
            for i in range(M_LEN):
                for j in range(N_LEN):
                    uoldtemp=uold[i,j,0]
                    voldtemp=vold[i,j,0]
                    poldtemp=pold[i,j,0] 
                    uold[i,j,0] = u[i,j,0] + alpha * (unew[i,j,0] - 2. * u[i,j,0] + uoldtemp)
                    vold[i,j,0] = v[i,j,0] + alpha * (vnew[i,j,0] - 2. * v[i,j,0] + voldtemp)
                    pold[i,j,0] = p[i,j,0] + alpha * (pnew[i,j,0] - 2. * p[i,j,0] + poldtemp)
    
            for i in range(M_LEN):
                    for j in range(N_LEN):
                        u[i,j,0] = unew[i,j,0]
                        v[i,j,0] = vnew[i,j,0]
                        p[i,j,0] = pnew[i,j,0]
    
        else:
            tdt = tdt+tdt
    
            uold = np.copy(u[...])
            vold = np.copy(v[...])
            pold = np.copy(p[...])
            u = np.copy(unew[...])
            v = np.copy(vnew[...])
            p = np.copy(pnew[...])
    
        if((VIS == True) & (ncycle%VIS_DT==0)):
            live_plot3(u, v, p, "ncycle: " + str(ncycle))
    # Print initial conditions
    if L_OUT:
           print("cycle number ", ITMAX)
           print(" diagonal elements of p:\n", pnew[:,:,0].diagonal()[:-1])
           print(" diagonal elements of u:\n", unew[:,:,0].diagonal()[:-1])
           print(" diagonal elements of v:\n", vnew[:,:,0].diagonal()[:-1])