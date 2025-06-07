"""
This version is close to swm_cartesian using 1 halo line per dimension.

e.g. for M=3, N=3, with 'x' = interior, '0' = periodic halo, the grid is:

for u
0 x x x
0 x x x
0 x x x
0 0 0 0

for v
0 0 0 0
x x x 0
x x x 0
x x x 0

for p
x x x 0
x x x 0
x x x 0
0 0 0 0
"""

from gt4py import next as gtx
from time import perf_counter
import initial_conditions
import utils
import config
from gt4py.next.otf import compiled_program

dtype = gtx.float64

backend = gtx.gtfn_gpu
# backend = gtx.gtfn_cpu
# backend = gtx.itir_python
# backend = None

print(f"Using backend '{getattr(backend, 'name', backend)}'.")

I = gtx.Dimension("I")
J = gtx.Dimension("J")

IJField = gtx.Field[gtx.Dims[I, J], dtype]


@gtx.field_operator
def _calc_cucvzh(
    u: IJField,
    v: IJField,
    p: IJField,
    fsdx: dtype,
    fsdy: dtype,
):
    cu = 0.5 * (p(I + 1) + p) * u
    cv = 0.5 * (p(J + 1) + p) * v
    z = (fsdx * (v(I + 1) - v) - fsdy * (u(J + 1) - u)) / (
        p(I + 1)(J + 1) + p(J + 1) + p + p(I + 1)
    )
    h = p + 0.25 * (u(I - 1) * u(I - 1) + u * u + v(J - 1) * v(J - 1) + v * v)
    return cu, cv, z, h


@gtx.program(backend=backend)
def calc_cucvzh(
    u: IJField,
    v: IJField,
    p: IJField,
    fsdx: dtype,
    fsdy: dtype,
    cu: IJField,
    cv: IJField,
    z: IJField,
    h: IJField,
    M: gtx.int32,
    N: gtx.int32,
):
    _calc_cucvzh(u, v, p, fsdx, fsdy, out=(cu, cv, z, h), domain={I: (0, M), J: (0, N)})


@gtx.field_operator
def _calc_uvp(
    tdts8: dtype,
    tdtsdx: dtype,
    tdtsdy: dtype,
    uold: IJField,
    vold: IJField,
    pold: IJField,
    cu: IJField,
    cv: IJField,
    z: IJField,
    h: IJField,
):
    unew = (
        uold
        + tdts8 * (z + z(J - 1)) * (cv(I + 1) + cv + cv(J - 1) + cv(I + 1)(J - 1))
        - tdtsdx * (h(I + 1) - h)
    )
    vnew = (
        vold
        - tdts8 * (z + z(I - 1)) * (cu(I - 1) + cu(I - 1)(J + 1) + cu + cu(J + 1))
        - tdtsdy * (h(J + 1) - h)
    )
    pnew = pold - tdtsdx * (cu - cu(I - 1)) - tdtsdy * (cv - cv(J - 1))
    return unew, vnew, pnew


@gtx.program(backend=backend)
def calc_uvp(
    tdts8: dtype,
    tdtsdx: dtype,
    tdtsdy: dtype,
    uold: IJField,
    vold: IJField,
    pold: IJField,
    cu: IJField,
    cv: IJField,
    z: IJField,
    h: IJField,
    unew: IJField,
    vnew: IJField,
    pnew: IJField,
    M: gtx.int32,
    N: gtx.int32,
):
    _calc_uvp(
        tdts8,
        tdtsdx,
        tdtsdy,
        uold,
        vold,
        pold,
        cu,
        cv,
        z,
        h,
        out=(unew, vnew, pnew),
        domain={I: (0, M), J: (0, N)},
    )


# TODO: this doesn't work because the 3 fields are on different domains
# we can shift them, but then the result will be on the same domain and we would have to hack the domain afterwards.
@gtx.field_operator
def _calc_uvp_old(
    alpha: dtype,
    v: IJField,
    vnew: IJField,
    vold: IJField,
    u: IJField,
    unew: IJField,
    uold: IJField,
    p: IJField,
    pnew: IJField,
    pold: IJField,
):
    uold_new = u + alpha * (unew - 2.0 * u + uold)
    vold_new = v + alpha * (vnew - 2.0 * v + vold)
    pold_new = p + alpha * (pnew - 2.0 * p + pold)
    return uold_new, vold_new, pold_new


@gtx.program(backend=backend)
def calc_uvp_old(
    alpha: dtype,
    v: IJField,
    vnew: IJField,
    vold: IJField,
    u: IJField,
    unew: IJField,
    uold: IJField,
    p: IJField,
    pnew: IJField,
    pold: IJField,
    M: gtx.int32,
    N: gtx.int32,
):
    _calc_uvp_old(
        alpha,
        v,
        vnew,
        vold,
        u,
        unew,
        uold,
        p,
        pnew,
        pold,
        out=(uold, vold, pold),
        domain={I: (0, M + 1), J: (0, N + 1)},
    )


@gtx.field_operator
def _update_var(alpha: dtype, var: IJField, var_new: IJField, var_old: IJField):
    return var + alpha * (var_new - 2.0 * var + var_old)


@gtx.program(backend=backend)
def update_var(
    alpha: dtype,
    var: IJField,
    var_new: IJField,
    var_old: IJField,
    istart: gtx.int32,
    iend: gtx.int32,
    jstart: gtx.int32,
    jend: gtx.int32,
):
    _update_var(
        alpha,
        var,
        var_new,
        var_old,
        out=var_old,
        domain={I: (istart, iend), J: (jstart, jend)},
    )


def main():
    dt0 = 0.0
    dt1 = 0.0
    dt15 = 0.0
    dt2 = 0.0
    dt25 = 0.0
    dt3 = 0.0

    M = config.M
    N = config.N

    _u, _v, _p = initial_conditions.initialize(M, N, config.dx, config.dy, config.a)

    u_domain = gtx.domain({I: (-1, M), J: (0, N + 1)})
    v_domain = gtx.domain({I: (0, M + 1), J: (-1, N)})
    p_domain = gtx.domain({I: (0, M + 1), J: (0, N + 1)})
    z_domain = gtx.domain({I: (-1, M), J: (-1, N)})

    def allocate(domain):
        return gtx.empty(domain, dtype=dtype, allocator=backend)

    h_gt = allocate(p_domain)
    z_gt = allocate(z_domain)
    cu_gt = allocate(u_domain)
    cv_gt = allocate(v_domain)
    pnew_gt = allocate(p_domain)
    unew_gt = allocate(u_domain)
    vnew_gt = allocate(v_domain)
    pold_gt = allocate(p_domain)
    uold_gt = allocate(u_domain)
    vold_gt = allocate(v_domain)

    u_gt = gtx.as_field(u_domain, _u, dtype=dtype, allocator=backend)
    v_gt = gtx.as_field(v_domain, _v, dtype=dtype, allocator=backend)
    p_gt = gtx.as_field(p_domain, _p, dtype=dtype, allocator=backend)

    # Save initial conditions
    uold_gt[...] = u_gt[...]
    vold_gt[...] = v_gt[...]
    pold_gt[...] = p_gt[...]

    # Print initial conditions
    if config.L_OUT:
        print(" Number of points in the x direction: ", M)
        print(" Number of points in the y direction: ", N)
        print(" grid spacing in the x direction: ", config.dx)
        print(" grid spacing in the y direction: ", config.dy)
        print(" time step: ", config.dt)
        print(" time filter coefficient: ", config.alpha)

        print(" Initial p:\n", p_gt[:, :].ndarray.diagonal()[:-1])
        print(" Initial u:\n", u_gt[:, :].ndarray.diagonal()[:-1])
        print(" Initial v:\n", v_gt[:, :].ndarray.diagonal()[:-1])

    calc_cucvzh.compile(offset_provider={})
    calc_uvp.compile(offset_provider={})
    calc_uvp_old.compile(offset_provider={})
    update_var.compile(offset_provider={})
    compiled_program._async_compilation_pool.shutdown(wait=True)

    t0_start = perf_counter()
    time = 0.0
    tdt = config.dt

    # Main time loop
    for ncycle in range(config.ITMAX):
        if (ncycle % 100 == 0) & (config.VIS == False):
            print(f"cycle number{ncycle}")

        if config.VAL_DEEP and ncycle <= 3:
            print("validating init")
            utils.validate_uvp(
                u_gt.asnumpy(),
                v_gt.asnumpy(),
                p_gt.asnumpy(),
                M,
                N,
                ncycle,
                "init",
            )

        t1_start = perf_counter()

        calc_cucvzh(
            u=u_gt,
            v=v_gt,
            p=p_gt,
            fsdx=config.fsdx,
            fsdy=config.fsdy,
            cu=cu_gt,
            cv=cv_gt,
            z=z_gt,
            h=h_gt,
            M=M,
            N=N,
            offset_provider={},
        )

        t1_stop = perf_counter()
        t15_start = perf_counter()
        dt1 = dt1 + (t1_stop - t1_start)

        t15_start = perf_counter()

        # Periodic Boundary conditions
        cu_gt[0, :] = cu_gt[M, :]

        h_gt[M, :] = h_gt[0, :]
        cv_gt[M, 1:] = cv_gt[0, 1:]
        z_gt[0, 1:] = z_gt[M, 1:]

        cv_gt[:, 0] = cv_gt[:, N]
        h_gt[:, N] = h_gt[:, 0]
        cu_gt[1:, N] = cu_gt[1:, 0]
        z_gt[1:, 0] = z_gt[1:, N]

        cu_gt[0, N] = cu_gt[M, 0]
        cv_gt[M, 0] = cv_gt[0, N]
        z_gt[0, 0] = z_gt[M, N]
        h_gt[M, N] = h_gt[0, 0]

        t15_stop = perf_counter()
        dt15 = dt15 + (t15_stop - t15_start)

        if config.VAL_DEEP and ncycle <= 1:
            utils.validate_cucvzh(
                cu_gt.asnumpy(),
                cv_gt.asnumpy(),
                z_gt.asnumpy(),
                h_gt.asnumpy(),
                M,
                N,
                ncycle,
                "t100",
            )

        # Calclulate new values of u,v, and p
        tdts8 = tdt / 8.0
        tdtsdx = tdt / config.dx
        tdtsdy = tdt / config.dy
        # print(tdts8, tdtsdx, tdtsdy)

        t2_start = perf_counter()

        calc_uvp(
            tdts8=tdts8,
            tdtsdx=tdtsdx,
            tdtsdy=tdtsdy,
            uold=uold_gt,
            vold=vold_gt,
            pold=pold_gt,
            cu=cu_gt,
            cv=cv_gt,
            z=z_gt,
            h=h_gt,
            unew=unew_gt,
            vnew=vnew_gt,
            pnew=pnew_gt,
            M=M,
            N=N,
            offset_provider={},
        )

        t2_stop = perf_counter()
        t25_start = perf_counter()
        dt2 = dt2 + (t2_stop - t2_start)

        # Periodic Boundary conditions
        unew_gt[0, :] = unew_gt[M, :]
        pnew_gt[M, :] = pnew_gt[0, :]
        vnew_gt[M, 1:] = vnew_gt[0, 1:]
        unew_gt[1:, N] = unew_gt[1:, 0]
        vnew_gt[:, 0] = vnew_gt[:, N]
        pnew_gt[:, N] = pnew_gt[:, 0]

        unew_gt[0, N] = unew_gt[M, 0]
        vnew_gt[M, 0] = vnew_gt[0, N]
        pnew_gt[M, N] = pnew_gt[0, 0]

        t25_stop = perf_counter()
        dt25 = dt25 + (t25_stop - t25_start)

        if config.VAL_DEEP and ncycle <= 1:
            utils.validate_uvp(
                unew_gt.asnumpy(),
                vnew_gt.asnumpy(),
                pnew_gt.asnumpy(),
                M,
                N,
                ncycle,
                "t200",
            )

        time = time + config.dt

        if ncycle > 0:
            t3_start = perf_counter()
            # ugly to get this in one kernel because of different domain
            update_var(
                alpha=config.alpha,
                var=u_gt,
                var_new=unew_gt,
                var_old=uold_gt,
                istart=-1,
                iend=M,
                jstart=0,
                jend=N + 1,
                offset_provider={},
            )
            update_var(
                alpha=config.alpha,
                var=v_gt,
                var_new=vnew_gt,
                var_old=vold_gt,
                istart=0,
                iend=M + 1,
                jstart=-1,
                jend=N,
                offset_provider={},
            )
            update_var(
                alpha=config.alpha,
                var=p_gt,
                var_new=pnew_gt,
                var_old=pold_gt,
                istart=0,
                iend=M + 1,
                jstart=0,
                jend=N + 1,
                offset_provider={},
            )
            # swap
            u_gt, unew_gt = unew_gt, u_gt
            v_gt, vnew_gt = vnew_gt, v_gt
            p_gt, pnew_gt = pnew_gt, p_gt

            t3_stop = perf_counter()
            dt3 = dt3 + (t3_stop - t3_start)

        else:
            tdt = tdt + tdt

            uold_gt[...] = u_gt[...]
            vold_gt[...] = v_gt[...]
            pold_gt[...] = p_gt[...]
            u_gt[...] = unew_gt[...]
            v_gt[...] = vnew_gt[...]
            p_gt[...] = pnew_gt[...]

        if (config.VIS) & (ncycle % config.VIS_DT == 0):
            utils.live_plot3(
                u_gt.asnumpy(),
                v_gt.asnumpy(),
                p_gt.asnumpy(),
                "ncycle: " + str(ncycle),
            )

    t0_stop = perf_counter()
    dt0 = dt0 + (t0_stop - t0_start)
    # Print initial conditions
    if config.L_OUT:
        print("cycle number ", config.ITMAX)
        print(" diagonal elements of p:\n", p_gt[:, :].ndarray.diagonal()[:-1])
        print(" diagonal elements of u:\n", u_gt[:, :].ndarray.diagonal()[:-1])
        print(" diagonal elements of v:\n", v_gt[:, :].ndarray.diagonal()[:-1])
    print("total: ", dt0)
    print("t100: ", dt1)
    print("t150: ", dt15)
    print("t200: ", dt2)
    print("t250: ", dt25)
    print("t300: ", dt3)

    if config.VAL:
        utils.final_validation(
            u_gt.asnumpy(),
            v_gt.asnumpy(),
            p_gt.asnumpy(),
            ITMAX=config.ITMAX,
            M=M,
            N=N,
        )


if __name__ == "__main__":
    main()
