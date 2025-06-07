"""
This version uses 2 halo lines (1 on each side)

e.g. for M=3, N=3, with 'x' = interior, '0' = periodic halo, the grid is:

for all fields
0 0 0 0 0
0 x x x 0
0 x x x 0
0 x x x 0
0 0 0 0 0
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


@gtx.field_operator
def _timestep(
    u: IJField,
    v: IJField,
    p: IJField,
    fsdx: dtype,
    fsdy: dtype,
    tdts8: dtype,
    tdtsdx: dtype,
    tdtsdy: dtype,
    uold: IJField,
    vold: IJField,
    pold: IJField,
    alpha: dtype,
):
    cu, cv, z, h = _calc_cucvzh(u, v, p, fsdx, fsdy)
    unew, vnew, pnew = _calc_uvp(tdts8, tdtsdx, tdtsdy, uold, vold, pold, cu, cv, z, h)
    uold, vold, pold = _calc_uvp_old(alpha, v, vnew, vold, u, unew, uold, p, pnew, pold)
    return (
        unew,
        vnew,
        pnew,
        uold,
        vold,
        pold,
    )


@gtx.program(backend=backend)
def timestep(
    u: IJField,
    v: IJField,
    p: IJField,
    fsdx: dtype,
    fsdy: dtype,
    tdts8: dtype,
    tdtsdx: dtype,
    tdtsdy: dtype,
    uold: IJField,
    vold: IJField,
    pold: IJField,
    alpha: dtype,
    unew: IJField,
    vnew: IJField,
    pnew: IJField,
    M: gtx.int32,
    N: gtx.int32,
):
    _timestep(
        u=u,
        v=v,
        p=p,
        fsdx=fsdx,
        fsdy=fsdy,
        tdts8=tdts8,
        tdtsdx=tdtsdx,
        tdtsdy=tdtsdy,
        uold=uold,
        vold=vold,
        pold=pold,
        alpha=alpha,
        out=(unew, vnew, pnew, uold, vold, pold),
        domain={I: (0, M), J: (0, N)},
    )


def apply_periodicity(x: IJField):
    """Apply periodicity to the field x."""
    x.ndarray[...] = x.array_ns.pad(
        x.ndarray[1:-1, 1:-1], ((1, 1), (1, 1)), mode="wrap"
    )
    return x


def main():
    dt0 = 0.0
    dt1 = 0.0
    dt15 = 0.0
    dt2 = 0.0
    dt25 = 0.0
    dt3 = 0.0

    M = config.M
    N = config.N

    _u, _v, _p = initial_conditions.initialize_2halo(
        M, N, config.dx, config.dy, config.a
    )

    domain = gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})

    def allocate(domain):
        return gtx.empty(domain, dtype=dtype, allocator=backend)

    pnew_gt = allocate(domain)
    unew_gt = allocate(domain)
    vnew_gt = allocate(domain)
    pold_gt = allocate(domain)
    uold_gt = allocate(domain)
    vold_gt = allocate(domain)

    u_gt = gtx.as_field(domain, _u, dtype=dtype, allocator=backend)
    v_gt = gtx.as_field(domain, _v, dtype=dtype, allocator=backend)
    p_gt = gtx.as_field(domain, _p, dtype=dtype, allocator=backend)

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

        print(" Initial p:\n", p_gt[:, :].ndarray.diagonal()[1:-1])
        print(" Initial u:\n", u_gt[:, :].ndarray.diagonal()[1:-1])
        print(" Initial v:\n", v_gt[:, :].ndarray.diagonal()[1:-1])

    if backend is not None:
        timestep.compile(offset_provider={})
        compiled_program._async_compilation_pool.shutdown(wait=True)

    t0_start = perf_counter()

    # Main time loop
    for ncycle in range(config.ITMAX):
        if (ncycle % 100 == 0) & (config.VIS == False):
            print(f"cycle number{ncycle}")

        if config.VAL_DEEP and ncycle <= 3:
            print("validating init")
            utils.validate_uvp(
                u_gt.asnumpy()[:-1, 1:],
                v_gt.asnumpy()[1:, :-1],
                p_gt.asnumpy()[1:, 1:],
                M,
                N,
                ncycle,
                "init",
            )

        tdt = config.dt if ncycle == 0 else config.dt * 2.0

        tdts8 = tdt / 8.0
        tdtsdx = tdt / config.dx
        tdtsdy = tdt / config.dy

        t3_start = perf_counter()
        timestep(
            u=u_gt,
            v=v_gt,
            p=p_gt,
            fsdx=config.fsdx,
            fsdy=config.fsdy,
            tdts8=tdts8,
            tdtsdx=tdtsdx,
            tdtsdy=tdtsdy,
            uold=uold_gt,
            vold=vold_gt,
            pold=pold_gt,
            alpha=config.alpha if ncycle > 0 else 0.0,
            unew=unew_gt,
            vnew=vnew_gt,
            pnew=pnew_gt,
            M=M,
            N=N,
            offset_provider={},
        )

        if hasattr(u_gt.array_ns, "cuda"):
            u_gt.array_ns.cuda.runtime.deviceSynchronize()
        t3_stop = perf_counter()
        dt3 = dt3 + (t3_stop - t3_start)

        # TODO add timer around the periodicity
        unew_gt = apply_periodicity(unew_gt)
        vnew_gt = apply_periodicity(vnew_gt)
        pnew_gt = apply_periodicity(pnew_gt)

        # swap x with xnew fields
        u_gt, unew_gt = unew_gt, u_gt
        v_gt, vnew_gt = vnew_gt, v_gt
        p_gt, pnew_gt = pnew_gt, p_gt

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
            u_gt.asnumpy()[:-1, 1:],
            v_gt.asnumpy()[1:, :-1],
            p_gt.asnumpy()[1:, 1:],
            ITMAX=config.ITMAX,
            M=M,
            N=N,
        )


if __name__ == "__main__":
    main()
