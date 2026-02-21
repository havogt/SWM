"""
Tests for the Array API shallow water model implementation.

Validates that the implementation:
  1. Produces correct results with numpy (against reference data)
  2. Produces correct results with jax.numpy (against reference data)
  3. Produces correct results with torch (against reference data)
  4. Passes array-api-strict compliance checking
  5. Works with jax.jit compilation
  6. Produces consistent results across all backends

Usage:
  python test_array_api.py
"""

import sys
import os
import numpy as np

# Ensure the swm_python directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swm_array_api


def _run_simulation(xp, M, N, ITMAX):
    """Run the shallow water simulation and return final fields."""
    dx = 100000.0
    dy = 100000.0
    a = 1000000.0
    dt = 90.0
    alpha = 0.001

    u, v, p = swm_array_api.initialize_2halo(xp, M, N, dx, dy, a)
    uold = xp.asarray(u, copy=True)
    vold = xp.asarray(v, copy=True)
    pold = xp.asarray(p, copy=True)

    for ncycle in range(ITMAX):
        tdt = dt if ncycle == 0 else dt * 2.0
        alpha_val = alpha if ncycle > 0 else 0.0

        unew, vnew, pnew, uold, vold, pold = swm_array_api.timestep(
            xp, u, v, p, uold, vold, pold, dx, dy, tdt, alpha_val, M, N
        )
        u = unew
        v = vnew
        p = pnew

    return u, v, p


def _to_reference_layout(u, v, p):
    """Convert 2-halo arrays to the reference (M+1, N+1) layout."""
    return u[:-1, 1:], v[1:, :-1], p[1:, 1:]


def _validate_against_reference(u_ref_layout, v_ref_layout, p_ref_layout, M, N, ITMAX):
    """Validate against binary reference data. Returns (uLinfN, vLinfN, pLinfN)."""
    u_file = f"../ref/{M}x{N}/u.step{ITMAX}.final.bin"
    v_file = f"../ref/{M}x{N}/v.step{ITMAX}.final.bin"
    p_file = f"../ref/{M}x{N}/p.step{ITMAX}.final.bin"
    u_ref = np.fromfile(u_file).reshape(M + 1, N + 1)
    v_ref = np.fromfile(v_file).reshape(M + 1, N + 1)
    p_ref = np.fromfile(p_file).reshape(M + 1, N + 1)

    uLinfN = np.linalg.norm(u_ref - u_ref_layout, np.inf)
    vLinfN = np.linalg.norm(v_ref - v_ref_layout, np.inf)
    pLinfN = np.linalg.norm(p_ref - p_ref_layout, np.inf)

    return uLinfN, vLinfN, pLinfN


def test_numpy():
    """Test that numpy produces correct results against reference data."""
    print("=" * 60)
    print("TEST: numpy backend")
    print("=" * 60)

    M, N, ITMAX = 16, 16, 4000
    u, v, p = _run_simulation(np, M, N, ITMAX)

    u_ref, v_ref, p_ref = _to_reference_layout(
        np.asarray(u), np.asarray(v), np.asarray(p)
    )
    uLinfN, vLinfN, pLinfN = _validate_against_reference(u_ref, v_ref, p_ref, M, N, ITMAX)

    print(f"  uLinfN: {uLinfN}")
    print(f"  vLinfN: {vLinfN}")
    print(f"  pLinfN: {pLinfN}")

    tol = 1e-6
    assert uLinfN < tol, f"u error too large: {uLinfN}"
    assert vLinfN < tol, f"v error too large: {vLinfN}"
    assert pLinfN < tol, f"p error too large: {pLinfN}"

    print("  PASSED\n")


def test_jax():
    """Test that jax.numpy produces correct results against reference data."""
    print("=" * 60)
    print("TEST: jax.numpy backend")
    print("=" * 60)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from array_api_compat import array_namespace

    M, N, ITMAX = 16, 16, 4000

    test_arr = jnp.zeros((1,))
    xp = array_namespace(test_arr)

    u, v, p = _run_simulation(xp, M, N, ITMAX)

    u_ref, v_ref, p_ref = _to_reference_layout(
        np.asarray(u), np.asarray(v), np.asarray(p)
    )
    uLinfN, vLinfN, pLinfN = _validate_against_reference(u_ref, v_ref, p_ref, M, N, ITMAX)

    print(f"  uLinfN: {uLinfN}")
    print(f"  vLinfN: {vLinfN}")
    print(f"  pLinfN: {pLinfN}")

    tol = 1e-6
    assert uLinfN < tol, f"u error too large: {uLinfN}"
    assert vLinfN < tol, f"v error too large: {vLinfN}"
    assert pLinfN < tol, f"p error too large: {pLinfN}"

    print("  PASSED\n")


def test_torch():
    """Test that torch produces correct results against reference data."""
    print("=" * 60)
    print("TEST: torch backend")
    print("=" * 60)

    import torch
    from array_api_compat import array_namespace

    M, N, ITMAX = 16, 16, 4000

    test_arr = torch.zeros((1,), dtype=torch.float64)
    xp = array_namespace(test_arr)

    u, v, p = _run_simulation(xp, M, N, ITMAX)

    u_ref, v_ref, p_ref = _to_reference_layout(
        np.asarray(u), np.asarray(v), np.asarray(p)
    )
    uLinfN, vLinfN, pLinfN = _validate_against_reference(u_ref, v_ref, p_ref, M, N, ITMAX)

    print(f"  uLinfN: {uLinfN}")
    print(f"  vLinfN: {vLinfN}")
    print(f"  pLinfN: {pLinfN}")

    tol = 1e-6
    assert uLinfN < tol, f"u error too large: {uLinfN}"
    assert vLinfN < tol, f"v error too large: {vLinfN}"
    assert pLinfN < tol, f"p error too large: {pLinfN}"

    print("  PASSED\n")


def test_strict_compliance():
    """Test that the implementation is compliant with the array API standard.

    Uses array_api_strict which raises errors for any non-standard operations.
    Runs a short simulation (10 steps) to verify compliance.
    """
    print("=" * 60)
    print("TEST: array-api-strict compliance")
    print("=" * 60)

    import array_api_strict
    array_api_strict.set_array_api_strict_flags(api_version="2024.12")

    M, N, ITMAX = 16, 16, 10
    u, v, p = _run_simulation(array_api_strict, M, N, ITMAX)

    print("  array_api_strict simulation completed without errors")
    print("  PASSED\n")


def test_jax_jit():
    """Test that the timestep function works under jax.jit compilation.

    Wraps the core timestep in jax.jit and verifies it produces the same
    results as the non-jitted version.
    """
    print("=" * 60)
    print("TEST: jax.jit compilation")
    print("=" * 60)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from array_api_compat import array_namespace

    M, N, ITMAX = 16, 16, 100

    test_arr = jnp.zeros((1,))
    xp = array_namespace(test_arr)

    dx = 100000.0
    dy = 100000.0
    a = 1000000.0
    dt_val = 90.0
    alpha = 0.001

    u, v, p = swm_array_api.initialize_2halo(xp, M, N, dx, dy, a)
    uold = xp.asarray(u, copy=True)
    vold = xp.asarray(v, copy=True)
    pold = xp.asarray(p, copy=True)

    # Create a jitted version of the timestep + halo update
    @jax.jit
    def jitted_timestep(u, v, p, uold, vold, pold, dt_val, alpha_val):
        return swm_array_api.timestep(
            xp, u, v, p, uold, vold, pold,
            dx, dy, dt_val, alpha_val, M, N
        )

    # Run simulation with jitted timestep
    for ncycle in range(ITMAX):
        tdt = dt_val if ncycle == 0 else dt_val * 2.0
        alpha_val = alpha if ncycle > 0 else 0.0

        unew, vnew, pnew, uold, vold, pold = jitted_timestep(
            u, v, p, uold, vold, pold, tdt, alpha_val
        )
        u = unew
        v = vnew
        p = pnew

    # Compare against non-jitted numpy reference
    u_np, v_np, p_np = _run_simulation(np, M, N, ITMAX)

    u_diff = np.max(np.abs(np.asarray(u_np) - np.asarray(u)))
    v_diff = np.max(np.abs(np.asarray(v_np) - np.asarray(v)))
    p_diff = np.max(np.abs(np.asarray(p_np) - np.asarray(p)))

    print(f"  u max diff vs numpy: {u_diff}")
    print(f"  v max diff vs numpy: {v_diff}")
    print(f"  p max diff vs numpy: {p_diff}")

    tol = 1e-10
    assert u_diff < tol, f"u difference too large: {u_diff}"
    assert v_diff < tol, f"v difference too large: {v_diff}"
    assert p_diff < tol, f"p difference too large: {p_diff}"

    print("  jax.jit compilation and execution successful")
    print("  PASSED\n")


def test_torch_compile():
    """Test that the timestep function works under torch.compile.

    Wraps the core timestep in torch.compile and verifies it produces the same
    results as the non-compiled version.
    """
    print("=" * 60)
    print("TEST: torch.compile")
    print("=" * 60)

    import torch
    from array_api_compat import array_namespace

    M, N, ITMAX = 16, 16, 100

    test_arr = torch.zeros((1,), dtype=torch.float64)
    xp = array_namespace(test_arr)

    dx = 100000.0
    dy = 100000.0
    a = 1000000.0
    dt_val = 90.0
    alpha = 0.001

    u, v, p = swm_array_api.initialize_2halo(xp, M, N, dx, dy, a)
    uold = xp.asarray(u, copy=True)
    vold = xp.asarray(v, copy=True)
    pold = xp.asarray(p, copy=True)

    # Create a compiled version of the timestep
    compiled_timestep = torch.compile(
        lambda u, v, p, uold, vold, pold, dt_val, alpha_val: swm_array_api.timestep(
            xp, u, v, p, uold, vold, pold,
            dx, dy, dt_val, alpha_val, M, N
        )
    )

    # Run simulation with compiled timestep
    for ncycle in range(ITMAX):
        tdt = dt_val if ncycle == 0 else dt_val * 2.0
        alpha_val = alpha if ncycle > 0 else 0.0

        unew, vnew, pnew, uold, vold, pold = compiled_timestep(
            u, v, p, uold, vold, pold, tdt, alpha_val
        )
        u = unew
        v = vnew
        p = pnew

    # Compare against non-compiled numpy reference
    u_np, v_np, p_np = _run_simulation(np, M, N, ITMAX)

    u_diff = np.max(np.abs(np.asarray(u_np) - np.asarray(u)))
    v_diff = np.max(np.abs(np.asarray(v_np) - np.asarray(v)))
    p_diff = np.max(np.abs(np.asarray(p_np) - np.asarray(p)))

    print(f"  u max diff vs numpy: {u_diff}")
    print(f"  v max diff vs numpy: {v_diff}")
    print(f"  p max diff vs numpy: {p_diff}")

    tol = 1e-10
    assert u_diff < tol, f"u difference too large: {u_diff}"
    assert v_diff < tol, f"v difference too large: {v_diff}"
    assert p_diff < tol, f"p difference too large: {p_diff}"

    print("  torch.compile compilation and execution successful")
    print("  PASSED\n")


def _to_numpy(arr):
    """Convert array to numpy, handling GPU/CUDA tensors."""
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(arr)


def test_torch_compile_cuda():
    """Test that torch.compile works on CUDA device.

    Runs a short simulation on CUDA with torch.compile and verifies
    results match the CPU numpy reference. Skipped if CUDA is not available.
    """
    print("=" * 60)
    print("TEST: torch.compile on CUDA")
    print("=" * 60)

    import torch
    if not torch.cuda.is_available():
        print("  SKIPPED: CUDA not available\n")
        return

    from array_api_compat import array_namespace

    M, N, ITMAX = 16, 16, 100

    torch.set_default_device("cuda")
    try:
        test_arr = torch.zeros((1,), dtype=torch.float64)
        xp = array_namespace(test_arr)

        dx = 100000.0
        dy = 100000.0
        a = 1000000.0
        dt_val = 90.0
        alpha = 0.001

        u, v, p = swm_array_api.initialize_2halo(xp, M, N, dx, dy, a)
        uold = xp.asarray(u, copy=True)
        vold = xp.asarray(v, copy=True)
        pold = xp.asarray(p, copy=True)

        compiled_timestep = torch.compile(
            lambda u, v, p, uold, vold, pold, dt_val, alpha_val: swm_array_api.timestep(
                xp, u, v, p, uold, vold, pold,
                dx, dy, dt_val, alpha_val, M, N
            )
        )

        for ncycle in range(ITMAX):
            tdt = dt_val if ncycle == 0 else dt_val * 2.0
            alpha_val = alpha if ncycle > 0 else 0.0

            unew, vnew, pnew, uold, vold, pold = compiled_timestep(
                u, v, p, uold, vold, pold, tdt, alpha_val
            )
            u = unew
            v = vnew
            p = pnew

        torch.cuda.synchronize()

        # Compare against CPU numpy reference
        u_np, v_np, p_np = _run_simulation(np, M, N, ITMAX)

        u_diff = np.max(np.abs(np.asarray(u_np) - _to_numpy(u)))
        v_diff = np.max(np.abs(np.asarray(v_np) - _to_numpy(v)))
        p_diff = np.max(np.abs(np.asarray(p_np) - _to_numpy(p)))

        print(f"  u max diff vs numpy: {u_diff}")
        print(f"  v max diff vs numpy: {v_diff}")
        print(f"  p max diff vs numpy: {p_diff}")

        tol = 1e-10
        assert u_diff < tol, f"u difference too large: {u_diff}"
        assert v_diff < tol, f"v difference too large: {v_diff}"
        assert p_diff < tol, f"p difference too large: {p_diff}"

        print("  torch.compile CUDA execution successful")
        print("  PASSED\n")
    finally:
        torch.set_default_device("cpu")


def test_jax_jit_gpu():
    """Test that jax.jit works on GPU device.

    Runs a short simulation on GPU with jax.jit and verifies results
    match the CPU numpy reference. Skipped if no GPU is available.
    """
    print("=" * 60)
    print("TEST: jax.jit on GPU")
    print("=" * 60)

    import jax
    jax.config.update("jax_enable_x64", True)

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        print("  SKIPPED: GPU not available for JAX\n")
        return

    if not gpu_devices:
        print("  SKIPPED: No GPU devices found for JAX\n")
        return

    import jax.numpy as jnp
    from array_api_compat import array_namespace

    M, N, ITMAX = 16, 16, 100
    gpu = gpu_devices[0]

    test_arr = jax.device_put(jnp.zeros((1,)), gpu)
    xp = array_namespace(test_arr)

    dx = 100000.0
    dy = 100000.0
    a = 1000000.0
    dt_val = 90.0
    alpha = 0.001

    u, v, p = swm_array_api.initialize_2halo(xp, M, N, dx, dy, a)
    # Move initial data to GPU
    u = jax.device_put(u, gpu)
    v = jax.device_put(v, gpu)
    p = jax.device_put(p, gpu)
    uold = jax.device_put(xp.asarray(u, copy=True), gpu)
    vold = jax.device_put(xp.asarray(v, copy=True), gpu)
    pold = jax.device_put(xp.asarray(p, copy=True), gpu)

    @jax.jit
    def jitted_timestep(u, v, p, uold, vold, pold, dt_val, alpha_val):
        return swm_array_api.timestep(
            xp, u, v, p, uold, vold, pold,
            dx, dy, dt_val, alpha_val, M, N
        )

    for ncycle in range(ITMAX):
        tdt = dt_val if ncycle == 0 else dt_val * 2.0
        alpha_val = alpha if ncycle > 0 else 0.0

        unew, vnew, pnew, uold, vold, pold = jitted_timestep(
            u, v, p, uold, vold, pold, tdt, alpha_val
        )
        u = unew
        v = vnew
        p = pnew

    u.block_until_ready()

    # Compare against CPU numpy reference
    u_np, v_np, p_np = _run_simulation(np, M, N, ITMAX)

    u_diff = np.max(np.abs(np.asarray(u_np) - np.asarray(u)))
    v_diff = np.max(np.abs(np.asarray(v_np) - np.asarray(v)))
    p_diff = np.max(np.abs(np.asarray(p_np) - np.asarray(p)))

    print(f"  u max diff vs numpy: {u_diff}")
    print(f"  v max diff vs numpy: {v_diff}")
    print(f"  p max diff vs numpy: {p_diff}")

    tol = 1e-10
    assert u_diff < tol, f"u difference too large: {u_diff}"
    assert v_diff < tol, f"v difference too large: {v_diff}"
    assert p_diff < tol, f"p difference too large: {p_diff}"

    print("  jax.jit GPU execution successful")
    print("  PASSED\n")


def test_cross_backend_consistency():
    """Test that numpy, jax, and torch produce identical results."""
    print("=" * 60)
    print("TEST: cross-backend consistency (numpy vs jax vs torch)")
    print("=" * 60)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from array_api_compat import array_namespace

    M, N, ITMAX = 16, 16, 100

    # numpy
    u_np, v_np, p_np = _run_simulation(np, M, N, ITMAX)

    # jax
    test_arr = jnp.zeros((1,))
    xp_jax = array_namespace(test_arr)
    u_jax, v_jax, p_jax = _run_simulation(xp_jax, M, N, ITMAX)

    # torch
    test_arr = torch.zeros((1,), dtype=torch.float64)
    xp_torch = array_namespace(test_arr)
    u_torch, v_torch, p_torch = _run_simulation(xp_torch, M, N, ITMAX)

    # Compare numpy vs jax
    u_diff_jax = np.max(np.abs(np.asarray(u_np) - np.asarray(u_jax)))
    v_diff_jax = np.max(np.abs(np.asarray(v_np) - np.asarray(v_jax)))
    p_diff_jax = np.max(np.abs(np.asarray(p_np) - np.asarray(p_jax)))

    print(f"  numpy vs jax:   u={u_diff_jax:.2e}  v={v_diff_jax:.2e}  p={p_diff_jax:.2e}")

    # Compare numpy vs torch
    u_diff_torch = np.max(np.abs(np.asarray(u_np) - np.asarray(u_torch)))
    v_diff_torch = np.max(np.abs(np.asarray(v_np) - np.asarray(v_torch)))
    p_diff_torch = np.max(np.abs(np.asarray(p_np) - np.asarray(p_torch)))

    print(f"  numpy vs torch: u={u_diff_torch:.2e}  v={v_diff_torch:.2e}  p={p_diff_torch:.2e}")

    tol = 1e-10
    for label, diff in [
        ("u numpy-jax", u_diff_jax), ("v numpy-jax", v_diff_jax), ("p numpy-jax", p_diff_jax),
        ("u numpy-torch", u_diff_torch), ("v numpy-torch", v_diff_torch), ("p numpy-torch", p_diff_torch),
    ]:
        assert diff < tol, f"{label} difference too large: {diff}"

    print("  PASSED\n")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    passed = 0
    failed = 0
    errors = []

    tests = [
        ("numpy validation", test_numpy),
        ("jax validation", test_jax),
        ("torch validation", test_torch),
        ("strict compliance", test_strict_compliance),
        ("jax.jit compilation", test_jax_jit),
        ("torch.compile", test_torch_compile),
        ("torch.compile CUDA", test_torch_compile_cuda),
        ("jax.jit GPU", test_jax_jit_gpu),
        ("cross-backend consistency", test_cross_backend_consistency),
    ]

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  FAILED: {e}\n")

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    if errors:
        for name, err in errors:
            print(f"  FAILED: {name}: {err}")
        sys.exit(1)
    else:
        print("  All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
