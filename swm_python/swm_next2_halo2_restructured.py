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

backend = None
backend = gtx.gtfn_gpu
# backend = gtx.gtfn_cpu
# backend = gtx.itir_python

print(f"Using backend '{getattr(backend, 'name', backend)}'.")

I = gtx.Dimension("I")
J = gtx.Dimension("J")

IJField = gtx.Field[gtx.Dims[I, J], dtype]


@gtx.field_operator
def avg_x(f: IJField):
    """Average field in the x direction."""
    return 0.5 * (f(I + 1) + f)


@gtx.field_operator
def avg_y(f: IJField):
    """Average field in the y direction."""
    return 0.5 * (f(J + 1) + f)


@gtx.field_operator
def avg_x_staggered(f: IJField):
    """Average field which is staggered in x in the x direction."""
    return 0.5 * (f(I - 1) + f)


@gtx.field_operator
def avg_y_staggered(f: IJField):
    """Average field which is staggered in y in the y direction."""
    return 0.5 * (f(J - 1) + f)


@gtx.field_operator
def delta_x(dx: dtype, f: IJField):
    """Calculate the difference in the x direction."""
    return (1.0 / dx) * (f(I + 1) - f)


@gtx.field_operator
def delta_y(dx: dtype, f: IJField):
    """Calculate the difference in the y direction."""
    return (1.0 / dx) * (f(J + 1) - f)


@gtx.field_operator
def delta_x_staggered(dx: dtype, f: IJField):
    """Calculate the difference in the x direction for field staggered in x."""
    return (1.0 / dx) * (f - f(I - 1))


@gtx.field_operator
def delta_y_staggered(dx: dtype, f: IJField):
    """Calculate the difference in the y direction for field staggered in y."""
    return (1.0 / dx) * (f - f(J - 1))


@gtx.field_operator
def _timestep(
    u: IJField,
    v: IJField,
    p: IJField,
    dx: dtype,
    dy: dtype,
    dt: dtype,
    uold: IJField,
    vold: IJField,
    pold: IJField,
    alpha: dtype,
) -> tuple[IJField, IJField, IJField, IJField, IJField, IJField]:
    cu = avg_x(p) * u
    cv = avg_y(p) * v
    z = (delta_x(dx, v) - delta_y(dy, u)) / avg_x(avg_y(p))
    h = p + 0.5 * (avg_x_staggered(u * u) + avg_y_staggered(v * v))

    unew = (
        uold
        + avg_y_staggered(z) * avg_y_staggered(avg_x(cv)) * dt
        - delta_x(dx, h) * dt
    )
    vnew = (
        vold
        - avg_x_staggered(z) * avg_x_staggered(avg_y(cu)) * dt
        - delta_y(dy, h) * dt
    )
    pnew = pold - delta_x_staggered(dx, cu) * dt - delta_y_staggered(dy, cv) * dt

    uold_new = u + alpha * (unew - 2.0 * u + uold)
    vold_new = v + alpha * (vnew - 2.0 * v + vold)
    pold_new = p + alpha * (pnew - 2.0 * p + pold)

    return (
        unew,
        vnew,
        pnew,
        uold_new,
        vold_new,
        pold_new,
    )


@gtx.program(backend=backend)
def timestep(
    u: IJField,
    v: IJField,
    p: IJField,
    dx: dtype,
    dy: dtype,
    dt: dtype,
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
        dx=dx,
        dy=dy,
        dt=dt,
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
    dt25 = 0.0
    dt3 = 0.0

    M = config.M
    N = config.N

    domain = gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})

    pnew = gtx.empty(domain, dtype=dtype, allocator=backend)
    unew = gtx.empty(domain, dtype=dtype, allocator=backend)
    vnew = gtx.empty(domain, dtype=dtype, allocator=backend)

    # Initialize fields
    _u, _v, _p = initial_conditions.initialize_2halo(
        M, N, config.dx, config.dy, config.a
    )
    u = gtx.as_field(domain, _u, dtype=dtype, allocator=backend)
    v = gtx.as_field(domain, _v, dtype=dtype, allocator=backend)
    p = gtx.as_field(domain, _p, dtype=dtype, allocator=backend)

    # Initial old fields
    uold = gtx.as_field(domain, _u, dtype=dtype, allocator=backend)
    vold = gtx.as_field(domain, _v, dtype=dtype, allocator=backend)
    pold = gtx.as_field(domain, _p, dtype=dtype, allocator=backend)

    # Print initial conditions
    if config.L_OUT:
        print(" Number of points in the x direction: ", M)
        print(" Number of points in the y direction: ", N)
        print(" grid spacing in the x direction: ", config.dx)
        print(" grid spacing in the y direction: ", config.dy)
        print(" time step: ", config.dt)
        print(" time filter coefficient: ", config.alpha)

        print(" Initial p:\n", p[:, :].ndarray.diagonal()[1:-1])
        print(" Initial u:\n", u[:, :].ndarray.diagonal()[1:-1])
        print(" Initial v:\n", v[:, :].ndarray.diagonal()[1:-1])

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
                u.asnumpy()[:-1, 1:],
                v.asnumpy()[1:, :-1],
                p.asnumpy()[1:, 1:],
                M,
                N,
                ncycle,
                "init",
            )

        t3_start = perf_counter()
        timestep(
            u=u,
            v=v,
            p=p,
            dx=config.dx,
            dy=config.dy,
            dt=config.dt if ncycle == 0 else config.dt * 2.0,
            uold=uold,
            vold=vold,
            pold=pold,
            alpha=config.alpha if ncycle > 0 else 0.0,
            unew=unew,
            vnew=vnew,
            pnew=pnew,
            M=M,
            N=N,
            offset_provider={},
        )

        if hasattr(u.array_ns, "cuda"):
            u.array_ns.cuda.runtime.deviceSynchronize()
        t3_stop = perf_counter()
        dt3 = dt3 + (t3_stop - t3_start)

        # TODO add timer around the periodicity
        t25_start = perf_counter()
        unew = apply_periodicity(unew)
        vnew = apply_periodicity(vnew)
        pnew = apply_periodicity(pnew)
        t25_stop = perf_counter()
        dt25 = dt25 + (t25_stop - t25_start)

        # swap x with xnew fields
        u, unew = unew, u
        v, vnew = vnew, v
        p, pnew = pnew, p

        if (config.VIS) & (ncycle % config.VIS_DT == 0):
            utils.live_plot3(
                u.asnumpy(),
                v.asnumpy(),
                p.asnumpy(),
                "ncycle: " + str(ncycle),
            )

    t0_stop = perf_counter()
    dt0 = dt0 + (t0_stop - t0_start)
    # Print initial conditions
    if config.L_OUT:
        print("cycle number ", config.ITMAX)
        print(" diagonal elements of p:\n", p[:, :].ndarray.diagonal()[:-1])
        print(" diagonal elements of u:\n", u[:, :].ndarray.diagonal()[:-1])
        print(" diagonal elements of v:\n", v[:, :].ndarray.diagonal()[:-1])
    print("total: ", dt0)
    print("t100+t200+t300: ", dt3)
    print("t150+t250: ", dt25)

    if config.VAL:
        utils.final_validation(
            u.asnumpy()[:-1, 1:],
            v.asnumpy()[1:, :-1],
            p.asnumpy()[1:, 1:],
            ITMAX=config.ITMAX,
            M=M,
            N=N,
        )


if __name__ == "__main__":
    main()
