import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Shallow Water Model")
    parser.add_argument('--M', type=int, default=256, help='Number of points in the x direction')
    parser.add_argument('--N', type=int, default=256, help='Number of points in the y direction')
    parser.add_argument('--L_OUT', type=bool, default=True, help='a boolean for L_OUT')

    
    args = parser.parse_args()
    print(args.M, args.N)
    
    # Initialize model parameters
    M = args.M
    N = args.N
    M_LEN = M + 1
    N_LEN = N + 1
    L_OUT = args.L_OUT
    ITMAX = 500
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
    di = tpi / M
    dj = tpi / N
    dtheta = tpi / M
    pcf = (pi * pi * a * a) / (el * el)
    SIZE = M_LEN * N_LEN

    # Model Variables
    u = np.zeros((M_LEN, N_LEN))
    v = np.zeros((M_LEN, N_LEN))
    p = np.zeros((M_LEN, N_LEN))
    unew = np.zeros((M_LEN, N_LEN))
    vnew = np.zeros((M_LEN, N_LEN))
    pnew = np.zeros((M_LEN, N_LEN))
    uold = np.zeros((M_LEN, N_LEN))
    vold = np.zeros((M_LEN, N_LEN))
    pold = np.zeros((M_LEN, N_LEN))
    cu = np.zeros((M_LEN, N_LEN))
    cv = np.zeros((M_LEN, N_LEN))
    z = np.zeros((M_LEN, N_LEN))
    h = np.zeros((M_LEN, N_LEN))
    psi = np.zeros((M_LEN, N_LEN))
    
    # Initial values of the stream function and p
    for i in range(M + 1):
        for j in range(N + 1):
            psi[i, j] = a * np.sin((i + .5) * di) * np.sin((j + .5) * dtheta)
            p[i, j] = pcf * (np.cos(2. * (i) * di) + np.cos(2. * (j) * dtheta)) + 50000.
            
    # Calculate initial u and v
    for i in range(M):
        for j in range(N):
            u[i+1,j] = -(psi[i+1,j+1] - psi[i+1,j]) / dy
            v[i,j+1] = (psi[i+1,j+1] - psi[i,j+1]) / dx
            
    # # Periodic Boundary conditions
    u[0, :] = u[M, :]
    v[M, 1:] = v[0, 1:]
    u[1:, N] = u[1:, 0]
    v[:, 0] = v[:, N]

    u[0, N] = u[M, 0]
    v[M, 0] = v[0, N]
    
    # Save initial conditions
    uold = np.copy(u)
    vold = np.copy(v)
    pold = np.copy(p)
    
    # Print initial conditions
    if L_OUT:
        print(" Number of points in the x direction: ", M)
        print(" Number of points in the y direction: ", N)
        print(" grid spacing in the x direction: ", dx)
        print(" grid spacing in the y direction: ", dy)
        print(" time step: ", dt)
        print(" time filter coefficient: ", alpha)
        print(" Initial p:\n", p.diagonal()[:-1])
        print(" Initial u:\n", u.diagonal()[:-1])
        print(" Initial v:\n", v.diagonal()[:-1])
    
    time = 0.0 
    # Main time loop
    for ncycle in range(ITMAX):
        if ncycle % 100 == 0:
            print("cycle number ", ncycle)
        # Calculate cu, cv, z, and h
        # for i in range(1, M):
        #     for j in range(N):
        #         cu[i, j] = .5 * (p[i, j] + p[i - 1, j]) * u[i, j]
        
        # for i in range(M):
        #     for j in range(1, N):
        #         cv[i, j] = .5 * (p[i, j] + p[i, j - 1]) * v[i, j]
                
        # for i in range(1, M):
        #     for j in range(1, N):
        #         z[i, j] = (fsdx * (v[i, j] - v[i - 1, j]) - 
        #                    fsdy * (u[i, j] - u[i, j - 1])
        #                    ) / (p[i - 1, j - 1] + p[i, j - 1] + p[i, j] + p[i - 1, j])
                
        # for i in range(M):
        #     for j in range(N):
        #         h[i, j] = p[i, j] + 0.25 * (u[i + 1, j] * u[i + 1, j] + u[i, j] * u[i, j] + 
        #                                     v[i, j + 1] * v[i, j + 1] + v[i, j] * v[i, j])
                
        for i in range(M):
            for j in range(N):
                cu[i + 1, j] = .5 * (p[i + 1, j] + p[i, j]) * u[i + 1, j]
                cv[i, j + 1] = .5 * (p[i, j + 1] + p[i, j]) * v[i, j + 1]
                z[i + 1, j + 1] = (fsdx * (v[i + 1, j + 1] - v[i, j + 1]) -
                                fsdy * (u[i + 1, j + 1] - u[i, j + 1] )
                                ) / (p[i, j] + p[i + 1, j] + p[i + 1, j + 1] + p[i, j + 1])
                h[i, j] = p[i, j] + 0.25 * (u[i + 1, j] * u[i + 1, j] + u[i, j] * u[i, j] +
                                        v[i, j + 1] * v[i, j + 1] + v[i, j] * v[i, j])
        
        # # Periodic Boundary conditions
        cu[0, :] = cu[M, :]
        h[M, :] = h[0, :]
        cv[M, 1:] = cv[0, 1:]
        z[0, 1:] = z[M, 1:]
        
        cv[:, 0] = cv[:, N]
        h[:, N] = h[:, 0]
        cu[1:, N] = cu[1:, 0]
        z[1:, N] = z[1:, 0]
            
        cu[0, N] = cu[M, 0]
        cv[M, 0] = cv[0, N]
        z[0, 0] = z[M, N]
        h[M, N] = h[0, 0]

        # Calclulate new values of u,v, and p
        tdts8 = tdt / 8.
        tdtsdx = tdt / dx
        tdtsdy = tdt / dy
        
        for i in range(M):
            for j in range(N):
                unew[i+1,j] = uold[i+1,j] + tdts8 * (z[i+1,j+1] + z[i+1,j]) * (cv[i+1,j+1] + cv[i+1,j] + cv[i,j+1] + cv[i,j]) - tdtsdx * (h[i+1,j] - h[i,j])
                vnew[i,j+1] = vold[i,j+1] - tdts8 * (z[i+1,j+1] + z[i,j+1]) * (cu[i+1,j+1] + cu[i+1,j] + cu[i,j+1] + cu[i,j]) - tdtsdy * (h[i,j+1] - h[i,j])
                pnew[i,j] = pold[i,j] - tdtsdx * (cu[i+1,j] - cu[i,j]) - tdtsdy * (cv[i,j+1] - cv[i,j])
        # for i in range(M):
        #     for j in range(N):
        #         unew[i + 1, j] = (uold[i + 1, j] + tdts8 * (z[i + 1, j + 1] + z[i + 1, j]) *
        #                         (cv[i + 1, j + 1] + cv[i + 1, j] + cv[i, j + 1] + cv[i, j]) -
        #                         tdtsdx * (h[i + 1, j] - h[i, j])
        #                         )
        # # for i in range(1, M):
        # #     for j in range(N):
        # #         unew[i, j] = (uold[i, j] + tdts8 * (z[i, j + 1] + z[i, j]) * 
        # #                       (cv[i, j + 1] + cv[i, j] + cv[i - 1, j + 1] + cv[i - 1, j]) - 
        # #                       tdtsdx * (h[i, j] - h[i - 1, j])
        # #                       )
        # for i in range(M):
        #     for j in range(N):        
        #         vnew[i, j + 1] = (vold[i, j + 1] - tdts8 * (z[i + 1, j + 1] + z[i, j + 1]) *
        #                         (cu[i + 1, j + 1] + cu[i + 1, j] + cu[i, j + 1] + cu[i, j]) -
        #                         tdtsdy * (h[i, j + 1] - h[i, j])
        #                         )
        # # for i in range(M):
        # #     for j in range(1, N):        
        # #         vnew[i, j] = (vold[i, j] - tdts8 * (z[i + 1, j] + z[i, j]) * 
        # #                       (cu[i + 1, j] + cu[i + 1, j - 1] + cu[i, j] + cu[i, j - 1]) - 
        # #                       tdtsdy * (h[i, j] - h[i, j - 1])
        # #                       )
        # for i in range(M):
        #     for j in range(N):
        #         pnew[i, j] = (pold[i, j] - tdtsdx * (cu[i + 1, j] - cu[i, j]) -
        #                         tdtsdy * (cv[i, j + 1] - cv[i, j])
        #                         )
                
        # Periodic Boundary conditions
        unew[0, :] = unew[M, :]
        pnew[M, :] = pnew[0, :]
        vnew[M, 1:] = vnew[0, 1:]
        unew[1:, N] = unew[1:, 0]
        vnew[:, 0] = vnew[:, N]
        pnew[:, N] = pnew[:, 0]
        
        unew[0, N] = unew[M, 0]
        vnew[M, 0] = vnew[0, N]
        pnew[M, N] = pnew[0, 0]
        
        time = time + dt

        if(ncycle > 1):
            for i in range(M_LEN):
                for j in range(N_LEN):
                    uold[i,j] = u[i,j] + alpha * (unew[i,j] - 2 * u[i,j] + uold[i,j])
                    vold[i,j] = v[i,j] + alpha * (vnew[i,j] - 2 * v[i,j] + vold[i,j])
                    pold[i,j] = p[i,j] + alpha * (pnew[i,j] - 2 * p[i,j] + pold[i,j])
                    
            for i in range(M_LEN):
                for j in range(N_LEN):
                    u[i,j] = unew[i,j]
                    v[i,j] = vnew[i,j]
                    p[i,j] = pnew[i,j]
        else:
            tdt = tdt + tdt
            uold = np.copy(u)
            vold = np.copy(v)
            pold = np.copy(p)
            u = np.copy(unew)
            v = np.copy(vnew)
            p = np.copy(pnew)
        
        # Print initial conditions
    if L_OUT:
        print("cycle number ", ITMAX)
        print(" diagonal elements of p:\n", pnew.diagonal()[:-1])
        print(" diagonal elements of u:\n", unew.diagonal()[:-1])
        print(" diagonal elements of v:\n", vnew.diagonal()[:-1])

if __name__ == "__main__":
    main()
