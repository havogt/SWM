"""
Shallow Water Model using the Python Array API standard.

This implementation uses symmetric halo lines (1 on each side) for all fields,
following the approach of swm_next2_halo2_restructured.py. The periodic boundary
conditions are applied via halo exchange rather than asymmetric padding.

Compatible with any array library supporting the Array API standard:
  numpy, jax.numpy, cupy, array_api_strict, etc.

Usage:
  python swm_array_api.py --array-library numpy
  python swm_array_api.py --array-library jax
  python swm_array_api.py --array-library torch
  python swm_array_api.py --array-library cupy
  python swm_array_api.py --strict             # validate compliance with array_api_strict wrapping
  python swm_array_api.py --array-library jax --compile    # run with jax.jit
  python swm_array_api.py --array-library torch --compile  # run with torch.compile
"""

import argparse
from time import perf_counter
from array_api_compat import array_namespace


def _get_array_module(name):
    """Import and return the array module for the given library name."""
    if name == "numpy":
        import numpy
        return numpy
    elif name == "jax":
        import jax.numpy
        return jax.numpy
    elif name == "torch":
        import torch
        return torch
    elif name == "cupy":
        import cupy
        return cupy
    elif name == "array_api_strict":
        import array_api_strict
        return array_api_strict
    else:
        raise ValueError(f"Unknown array library: {name}")


def initialize_interior(xp, M, N, dx, dy, a):
    """Create initial u, v, p fields on the interior (M x N) grid."""
    pi = 4.0 * xp.atan(xp.asarray(1.0, dtype=xp.float64))
    tpi = 2.0 * pi
    d_i = tpi / M
    d_j = tpi / N
    el = N * dx
    pcf = (pi * pi * a * a) / (el * el)

    i_vals = xp.arange(0, M + 1, dtype=xp.float64)
    j_vals = xp.arange(0, N + 1, dtype=xp.float64)
    i_interior = xp.arange(0, M, dtype=xp.float64)
    j_interior = xp.arange(0, N, dtype=xp.float64)

    # psi: (M+1) x (N+1), p: (M) x (N), u: (M) x (N), v: (M) x (N)
    # Use reshape to create 2D broadcasting: column * row
    i_col = xp.reshape(i_vals, (M + 1, 1))  # (M+1, 1)
    j_row = xp.reshape(j_vals, (1, N + 1))  # (1, N+1)
    psi = a * xp.sin((i_col + 0.5) * d_i) * xp.sin((j_row + 0.5) * d_j)

    i_int_col = xp.reshape(i_interior, (M, 1))  # (M, 1)
    j_int_row = xp.reshape(j_interior, (1, N))  # (1, N)
    p = pcf * (xp.cos(2.0 * i_int_col * d_i) + xp.cos(2.0 * j_int_row * d_j)) + 50000.0

    u = -(psi[1:, 1:] - psi[1:, :-1]) / dy
    v = (psi[1:, 1:] - psi[:-1, 1:]) / dx

    return u, v, p


def apply_periodic_halo(xp, interior, x):
    """Apply periodic boundary conditions by filling the halo from the interior.

    The array x has shape (M+2, N+2) where the interior is x[1:-1, 1:-1].
    The halos are filled by wrapping around the interior periodically.

    Args:
        xp: array namespace
        interior: slice for the interior, just used for documentation
        x: the field array of shape (M+2, N+2)

    Returns:
        New array with halos filled periodically.
    """
    M_plus_2 = x.shape[0]
    N_plus_2 = x.shape[1]
    M = M_plus_2 - 2
    N = N_plus_2 - 2

    # Build the periodically-wrapped array from scratch.
    # Interior rows: x[1:-1, :] with wrapped columns
    # Then wrap top/bottom rows.

    # Extract the interior
    core = x[1:-1, 1:-1]  # (M, N)

    # Build columns: [last_col | core | first_col]
    left_col = core[:, N - 1:N]  # (M, 1) - last column wraps to left halo
    right_col = core[:, 0:1]      # (M, 1) - first column wraps to right halo
    middle_rows = xp.concat([left_col, core, right_col], axis=1)  # (M, N+2)

    # Build top and bottom halo rows from middle_rows
    top_row = middle_rows[M - 1:M, :]    # (1, N+2) - last row wraps to top
    bottom_row = middle_rows[0:1, :]      # (1, N+2) - first row wraps to bottom

    result = xp.concat([top_row, middle_rows, bottom_row], axis=0)  # (M+2, N+2)
    return result


def timestep(xp, u, v, p, uold, vold, pold, dx, dy, dt_val, alpha_val, M, N):
    """Perform one timestep of the shallow water equations.

    All fields have shape (M+2, N+2) with 1-wide symmetric halos.
    The computation domain is [0:M, 0:N] in the interior (indices [1:-1, 1:-1]).
    With halo data available, stencil operations can read neighbors without bounds issues.

    Following swm_next2_halo2_restructured.py, the stencil operations are:
      cu = avg_x(p) * u           = 0.5*(p[i+1,j] + p[i,j]) * u[i,j]
      cv = avg_y(p) * v           = 0.5*(p[i,j+1] + p[i,j]) * v[i,j]
      z  = (delta_x(v) - delta_y(u)) / avg_x(avg_y(p))
      h  = p + 0.5*(avg_x_staggered(u*u) + avg_y_staggered(v*v))

      unew = uold + avg_y_staggered(z)*avg_y_staggered(avg_x(cv))*dt - delta_x(h)*dt
      vnew = vold - avg_x_staggered(z)*avg_x_staggered(avg_y(cu))*dt - delta_y(h)*dt
      pnew = pold - delta_x_staggered(cu)*dt - delta_y_staggered(cv)*dt

    Where:
      avg_x(f)[i,j]          = 0.5*(f[i+1,j] + f[i,j])
      avg_y(f)[i,j]          = 0.5*(f[i,j+1] + f[i,j])
      avg_x_staggered(f)[i,j]= 0.5*(f[i-1,j] + f[i,j])
      avg_y_staggered(f)[i,j]= 0.5*(f[i,j-1] + f[i,j])
      delta_x(f)[i,j]        = (1/dx)*(f[i+1,j] - f[i,j])
      delta_y(f)[i,j]        = (1/dy)*(f[i,j+1] - f[i,j])
      delta_x_staggered(f)[i,j] = (1/dx)*(f[i,j] - f[i-1,j])
      delta_y_staggered(f)[i,j] = (1/dy)*(f[i,j] - f[i,j-1])

    Using the halo layout, if interior is [1:M+1, 1:N+1], then for a point i,j
    in the interior, i+1 and i-1 and j+1 and j-1 are all valid array indices.
    """
    # Slice aliases for readability (operating on the full (M+2, N+2) array)
    # Interior points: [1:M+1, 1:N+1]
    # We compute on interior and use neighbors via shifted slices.

    # -- Step 1: compute intermediate fields cu, cv, z, h --

    # avg_x(p) = 0.5*(p[i+1,j] + p[i,j]) for interior i in [1..M], j in [1..N]
    # cu = avg_x(p) * u
    cu_interior = 0.5 * (p[2:M + 2, 1:N + 1] + p[1:M + 1, 1:N + 1]) * u[1:M + 1, 1:N + 1]

    # avg_y(p) = 0.5*(p[i,j+1] + p[i,j])
    # cv = avg_y(p) * v
    cv_interior = 0.5 * (p[1:M + 1, 2:N + 2] + p[1:M + 1, 1:N + 1]) * v[1:M + 1, 1:N + 1]

    # z = (delta_x(v) - delta_y(u)) / avg_x(avg_y(p))
    # delta_x(v) = (1/dx)*(v[i+1,j] - v[i,j])
    # delta_y(u) = (1/dy)*(u[i,j+1] - u[i,j])
    # avg_x(avg_y(p)) = avg_x(0.5*(p[i,j+1]+p[i,j]))
    #                  = 0.5*(0.5*(p[i+1,j+1]+p[i+1,j]) + 0.5*(p[i,j+1]+p[i,j]))
    #                  = 0.25*(p[i,j] + p[i+1,j] + p[i+1,j+1] + p[i,j+1])
    delta_x_v = (1.0 / dx) * (v[2:M + 2, 1:N + 1] - v[1:M + 1, 1:N + 1])
    delta_y_u = (1.0 / dy) * (u[1:M + 1, 2:N + 2] - u[1:M + 1, 1:N + 1])
    avg_xy_p = 0.25 * (
        p[1:M + 1, 1:N + 1]
        + p[2:M + 2, 1:N + 1]
        + p[2:M + 2, 2:N + 2]
        + p[1:M + 1, 2:N + 2]
    )
    z_interior = (delta_x_v - delta_y_u) / avg_xy_p

    # h = p + 0.5*(avg_x_staggered(u*u) + avg_y_staggered(v*v))
    # avg_x_staggered(u*u) = 0.5*(u[i-1,j]^2 + u[i,j]^2)
    # avg_y_staggered(v*v) = 0.5*(v[i,j-1]^2 + v[i,j]^2)
    uu = u * u
    vv = v * v
    avg_xs_uu = 0.5 * (uu[0:M, 1:N + 1] + uu[1:M + 1, 1:N + 1])
    avg_ys_vv = 0.5 * (vv[1:M + 1, 0:N] + vv[1:M + 1, 1:N + 1])
    h_interior = p[1:M + 1, 1:N + 1] + 0.5 * (avg_xs_uu + avg_ys_vv)

    # Embed cu, cv, z, h into (M+2, N+2) arrays and apply periodicity
    cu_full = xp.zeros_like(u)
    cv_full = xp.zeros_like(v)
    z_full = xp.zeros_like(u)
    h_full = xp.zeros_like(p)

    # Set interior values via constructing new arrays
    # Since some array APIs don't support item assignment, we build from scratch
    cu_full = _set_interior(xp, cu_full, cu_interior, M, N)
    cv_full = _set_interior(xp, cv_full, cv_interior, M, N)
    z_full = _set_interior(xp, z_full, z_interior, M, N)
    h_full = _set_interior(xp, h_full, h_interior, M, N)

    # Apply periodic halos to intermediate fields
    cu_full = apply_periodic_halo(xp, None, cu_full)
    cv_full = apply_periodic_halo(xp, None, cv_full)
    z_full = apply_periodic_halo(xp, None, z_full)
    h_full = apply_periodic_halo(xp, None, h_full)

    # -- Step 2: compute new u, v, p --

    # unew = uold + avg_y_staggered(z)*avg_y_staggered(avg_x(cv))*dt - delta_x(h)*dt
    # avg_y_staggered(z) = 0.5*(z[i,j-1] + z[i,j])
    avg_ys_z = 0.5 * (z_full[1:M + 1, 0:N] + z_full[1:M + 1, 1:N + 1])
    # avg_x(cv) = 0.5*(cv[i+1,j] + cv[i,j])
    avg_x_cv = 0.5 * (cv_full[2:M + 2, 1:N + 1] + cv_full[1:M + 1, 1:N + 1])
    # But we need avg_y_staggered(avg_x(cv)) -- need avg_x(cv) with halo too
    # Let's compute avg_x(cv) on full domain first, then take staggered avg
    avg_x_cv_full = xp.zeros_like(u)
    avg_x_cv_interior = 0.5 * (cv_full[2:M + 2, 1:N + 1] + cv_full[1:M + 1, 1:N + 1])
    avg_x_cv_full = _set_interior(xp, avg_x_cv_full, avg_x_cv_interior, M, N)
    avg_x_cv_full = apply_periodic_halo(xp, None, avg_x_cv_full)
    avg_ys_avg_x_cv = 0.5 * (avg_x_cv_full[1:M + 1, 0:N] + avg_x_cv_full[1:M + 1, 1:N + 1])

    # delta_x(h) = (1/dx)*(h[i+1,j] - h[i,j])
    delta_x_h = (1.0 / dx) * (h_full[2:M + 2, 1:N + 1] - h_full[1:M + 1, 1:N + 1])

    unew_interior = (
        uold[1:M + 1, 1:N + 1]
        + avg_ys_z * avg_ys_avg_x_cv * dt_val
        - delta_x_h * dt_val
    )

    # vnew = vold - avg_x_staggered(z)*avg_x_staggered(avg_y(cu))*dt - delta_y(h)*dt
    avg_xs_z = 0.5 * (z_full[0:M, 1:N + 1] + z_full[1:M + 1, 1:N + 1])
    # avg_y(cu) on full domain
    avg_y_cu_full = xp.zeros_like(u)
    avg_y_cu_interior = 0.5 * (cu_full[1:M + 1, 2:N + 2] + cu_full[1:M + 1, 1:N + 1])
    avg_y_cu_full = _set_interior(xp, avg_y_cu_full, avg_y_cu_interior, M, N)
    avg_y_cu_full = apply_periodic_halo(xp, None, avg_y_cu_full)
    avg_xs_avg_y_cu = 0.5 * (avg_y_cu_full[0:M, 1:N + 1] + avg_y_cu_full[1:M + 1, 1:N + 1])

    # delta_y(h) = (1/dy)*(h[i,j+1] - h[i,j])
    delta_y_h = (1.0 / dy) * (h_full[1:M + 1, 2:N + 2] - h_full[1:M + 1, 1:N + 1])

    vnew_interior = (
        vold[1:M + 1, 1:N + 1]
        - avg_xs_z * avg_xs_avg_y_cu * dt_val
        - delta_y_h * dt_val
    )

    # pnew = pold - delta_x_staggered(cu)*dt - delta_y_staggered(cv)*dt
    delta_xs_cu = (1.0 / dx) * (cu_full[1:M + 1, 1:N + 1] - cu_full[0:M, 1:N + 1])
    delta_ys_cv = (1.0 / dy) * (cv_full[1:M + 1, 1:N + 1] - cv_full[1:M + 1, 0:N])

    pnew_interior = (
        pold[1:M + 1, 1:N + 1] - delta_xs_cu * dt_val - delta_ys_cv * dt_val
    )

    # Build full arrays with halos
    unew = _build_with_halo(xp, u, unew_interior, M, N)
    vnew = _build_with_halo(xp, v, vnew_interior, M, N)
    pnew = _build_with_halo(xp, p, pnew_interior, M, N)

    # -- Step 3: time filter (update old fields) --
    uold_new_interior = (
        u[1:M + 1, 1:N + 1]
        + alpha_val * (unew[1:M + 1, 1:N + 1] - 2.0 * u[1:M + 1, 1:N + 1] + uold[1:M + 1, 1:N + 1])
    )
    vold_new_interior = (
        v[1:M + 1, 1:N + 1]
        + alpha_val * (vnew[1:M + 1, 1:N + 1] - 2.0 * v[1:M + 1, 1:N + 1] + vold[1:M + 1, 1:N + 1])
    )
    pold_new_interior = (
        p[1:M + 1, 1:N + 1]
        + alpha_val * (pnew[1:M + 1, 1:N + 1] - 2.0 * p[1:M + 1, 1:N + 1] + pold[1:M + 1, 1:N + 1])
    )

    uold_new = _build_with_halo(xp, uold, uold_new_interior, M, N)
    vold_new = _build_with_halo(xp, vold, vold_new_interior, M, N)
    pold_new = _build_with_halo(xp, pold, pold_new_interior, M, N)

    return unew, vnew, pnew, uold_new, vold_new, pold_new


def _set_interior(xp, full_arr, interior, M, N):
    """Return a new array where the interior [1:M+1, 1:N+1] is set to `interior`.

    Uses only standard Array API operations (concat/stack) to avoid in-place mutation.
    """
    # Build the result row by row using concat.
    # Top halo row (row 0): zeros
    top_row = full_arr[0:1, :]  # (1, N+2)
    # Bottom halo row (row M+1): zeros
    bottom_row = full_arr[M + 1:M + 2, :]  # (1, N+2)

    # Middle rows: [halo_left | interior_row | halo_right]
    left_col = full_arr[1:M + 1, 0:1]    # (M, 1) - left halo zeros
    right_col = full_arr[1:M + 1, N + 1:N + 2]  # (M, 1) - right halo zeros
    middle = xp.concat([left_col, interior, right_col], axis=1)  # (M, N+2)

    return xp.concat([top_row, middle, bottom_row], axis=0)  # (M+2, N+2)


def _build_with_halo(xp, template, interior, M, N):
    """Build a full (M+2, N+2) array from interior values and apply periodic halo."""
    full = xp.zeros_like(template)
    full = _set_interior(xp, full, interior, M, N)
    return apply_periodic_halo(xp, None, full)


def initialize_2halo(xp, M, N, dx, dy, a):
    """Initialize fields with 2-halo (1 on each side) symmetric padding."""
    u, v, p = initialize_interior(xp, M, N, dx, dy, a)
    # Apply periodic halo by building (M+2, N+2) arrays
    u_full = _build_with_halo(xp, xp.zeros((M + 2, N + 2), dtype=xp.float64), u, M, N)
    v_full = _build_with_halo(xp, xp.zeros((M + 2, N + 2), dtype=xp.float64), v, M, N)
    p_full = _build_with_halo(xp, xp.zeros((M + 2, N + 2), dtype=xp.float64), p, M, N)
    return u_full, v_full, p_full


def to_reference_layout(arr, M, N):
    """Convert from 2-halo (M+2, N+2) layout to reference (M+1, N+1) layout.

    The reference data uses an asymmetric layout where:
      u: padded with (1,0) in x and (0,1) in y  -> u_ref = [halo; interior_rows][interior_cols; halo]
      v: padded with (0,1) in x and (1,0) in y
      p: padded with (0,1) in x and (0,1) in y

    For the 2-halo symmetric layout, the interior is at [1:M+1, 1:N+1].
    The reference format stores M+1 x N+1 values.

    For u: ref has rows [M, 0..M-1] and cols [0..N-1, 0] -> u_ref = u_2halo[0:M+1, 1:N+2]
      which is u_2halo[:-1, 1:]
    For v: ref has rows [0..N-1, 0] and cols [N, 0..N-1] -> v_ref = u_2halo[1:M+2, 0:N+1]
      which is v_2halo[1:, :-1]
    For p: ref has rows [0..M-1, 0] and cols [0..N-1, 0] -> p_ref = p_2halo[1:M+2, 1:N+2]
      which is p_2halo[1:, 1:]
    """
    pass  # implemented inline in validation


def main():
    parser = argparse.ArgumentParser(description="Shallow Water Model (Array API)")
    parser.add_argument(
        "--array-library",
        type=str,
        default="numpy",
        choices=["numpy", "jax", "torch", "cupy", "array_api_strict"],
        help="Array library to use",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable array-api-strict compliance checking via array_api_compat",
    )
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--ITMAX", type=int, default=4000)
    parser.add_argument("--validate", action="store_true", help="Validate against reference data")
    parser.add_argument("--validate-deep", action="store_true", help="Deep validation of each step")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable JIT compilation (jax.jit for jax, torch.compile for torch)",
    )
    parser.add_argument("--no-output", action="store_true", help="Suppress diagnostic output")
    args = parser.parse_args()

    M = args.M
    N = args.N
    ITMAX = args.ITMAX
    L_OUT = not args.no_output

    # Physical parameters
    dt = 90.0
    dx = 100000.0
    dy = 100000.0
    a = 1000000.0
    alpha = 0.001

    # Get the array module
    if args.array_library == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)

    lib = _get_array_module(args.array_library)

    if args.strict:
        import array_api_strict
        array_api_strict.set_array_api_strict_flags(api_version="2024.12")

    if args.strict and args.array_library != "array_api_strict":
        # Wrap the library arrays in array_api_strict for compliance testing
        import array_api_strict
        xp = array_api_strict
        print(f"Running with {args.array_library} arrays wrapped in array_api_strict for compliance checking")
    elif args.strict:
        import array_api_strict
        xp = array_api_strict
        print("Running with array_api_strict directly")
    else:
        xp = lib
        # Get the array-api-compat namespace for the library
        test_arr = lib.zeros((1,))
        xp = array_namespace(test_arr)
        print(f"Running with {args.array_library} via array_api_compat namespace")

    # Initialize fields
    u, v, p = initialize_2halo(xp, M, N, dx, dy, a)

    if args.strict and args.array_library != "array_api_strict":
        # Convert to array_api_strict arrays
        import array_api_strict
        import numpy as np
        u_np = np.asarray(u) if hasattr(u, '__array__') else u
        v_np = np.asarray(v) if hasattr(v, '__array__') else v
        p_np = np.asarray(p) if hasattr(p, '__array__') else p
        u = array_api_strict.asarray(u_np, dtype=array_api_strict.float64)
        v = array_api_strict.asarray(v_np, dtype=array_api_strict.float64)
        p = array_api_strict.asarray(p_np, dtype=array_api_strict.float64)
        xp = array_api_strict

    uold = xp.asarray(u, copy=True)
    vold = xp.asarray(v, copy=True)
    pold = xp.asarray(p, copy=True)

    if L_OUT:
        print(f" Number of points in the x direction: {M}")
        print(f" Number of points in the y direction: {N}")
        print(f" grid spacing in the x direction: {dx}")
        print(f" grid spacing in the y direction: {dy}")
        print(f" time step: {dt}")
        print(f" time filter coefficient: {alpha}")

    # For validation, we need numpy
    if args.validate or args.validate_deep:
        import numpy as np

    if args.validate_deep:
        import sys
        sys.path.insert(0, "/home/user/SWM/swm_python")
        import utils

    # Set up the timestep function, optionally with JIT compilation
    def timestep_fn(u, v, p, uold, vold, pold, dt_val, alpha_val):
        return timestep(xp, u, v, p, uold, vold, pold, dx, dy, dt_val, alpha_val, M, N)

    if args.compile:
        if args.array_library == "jax":
            import jax
            timestep_fn = jax.jit(timestep_fn)
            print("JIT compilation enabled via jax.jit")
        elif args.array_library == "torch":
            import torch
            timestep_fn = torch.compile(timestep_fn)
            print("JIT compilation enabled via torch.compile")
        else:
            print(f"Warning: --compile has no effect for {args.array_library}")

    dt_total = 0.0
    dt_compute = 0.0

    t0_start = perf_counter()

    for ncycle in range(ITMAX):
        if ncycle % 100 == 0:
            print(f"cycle number {ncycle}")

        if args.validate_deep and ncycle <= 3:
            import numpy as np
            u_np = np.asarray(u)
            v_np = np.asarray(v)
            p_np = np.asarray(p)
            # Convert 2-halo to reference layout
            utils.validate_uvp(
                u_np[:-1, 1:], v_np[1:, :-1], p_np[1:, 1:],
                M, N, ncycle, "init",
            )

        tdt = dt if ncycle == 0 else dt * 2.0
        alpha_val = alpha if ncycle > 0 else 0.0

        t_start = perf_counter()
        unew, vnew, pnew, uold, vold, pold = timestep_fn(
            u, v, p, uold, vold, pold, tdt, alpha_val
        )
        t_stop = perf_counter()
        dt_compute += t_stop - t_start

        u = unew
        v = vnew
        p = pnew

    t0_stop = perf_counter()
    dt_total = t0_stop - t0_start

    if L_OUT:
        print(f"cycle number {ITMAX}")

    print(f"total: {dt_total}")
    print(f"compute: {dt_compute}")

    if args.validate:
        import numpy as np
        u_np = np.asarray(u)
        v_np = np.asarray(v)
        p_np = np.asarray(p)

        # Convert to reference layout for validation
        u_ref_layout = u_np[:-1, 1:]
        v_ref_layout = v_np[1:, :-1]
        p_ref_layout = p_np[1:, 1:]

        sys_path_added = False
        import sys
        if "/home/user/SWM/swm_python" not in sys.path:
            sys.path.insert(0, "/home/user/SWM/swm_python")
            sys_path_added = True
        import utils
        utils.final_validation(u_ref_layout, v_ref_layout, p_ref_layout, ITMAX=ITMAX, M=M, N=N)


if __name__ == "__main__":
    main()
