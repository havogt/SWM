# Automatic Differentiation for GT4Py: Project Plan & Background

## 1. Overview: Where and How AD Is Used in Weather & Climate

Automatic differentiation (AD) computes exact derivatives of a program's outputs with respect to its inputs — not symbolically, and not by finite differences, but by systematically applying the chain rule through the computation graph. In weather and climate science, this capability unlocks several major application areas.

### 1.1 Data Assimilation (4D-Var)

This is the single most important historical use of AD in weather/climate. Every operational weather forecast starts from an "analysis" — the best estimate of the current atmospheric state. 4D-Var data assimilation finds this optimal initial condition by minimizing a cost function:

```
J(x₀) = ½(x₀ - xᵇ)ᵀ B⁻¹ (x₀ - xᵇ)  +  ½ Σₖ (H[M(x₀, tₖ)] - yₖ)ᵀ R⁻¹ (H[M(x₀, tₖ)] - yₖ)
```

where:
- `x₀` is the initial state we're optimizing
- `xᵇ` is the background (prior) estimate, `B` its error covariance
- `M(x₀, tₖ)` is the model forecast from `x₀` to time `tₖ`
- `H` is the observation operator (maps model state to observation space)
- `yₖ` are actual observations at time `tₖ`, `R` their error covariance

To minimize `J` efficiently with gradient-based methods (e.g., L-BFGS), we need `∇J` with respect to `x₀`. The initial state `x₀` is *huge* (millions of degrees of freedom), so we need the gradient with respect to all these variables from a single scalar output — this is exactly what **reverse-mode AD (the adjoint)** provides.

ECMWF (the world's leading weather forecasting center) uses 4D-Var operationally. Historically, building the tangent-linear and adjoint models was a massive manual effort that took years. AD tools like TAF/Tapenade have automated parts of this, and JAX-based differentiable models eliminate the problem entirely.

### 1.2 Hybrid AI-Physics Models (e.g., NeuralGCM)

This is the exciting modern frontier. The idea: take a physics-based dynamical core (solving the equations of fluid motion) and replace or augment uncertain sub-grid parameterizations (clouds, convection, turbulence) with neural networks. The entire coupled system — physics solver + neural network — is made differentiable so the neural network can be trained "online" (end-to-end) by backpropagating through the physics.

**Why this matters:** Traditional "offline" training learns parameterizations independently of the dynamics. When plugged back into the model, they often cause instability and climate drift. End-to-end training through a differentiable solver avoids this because the neural network learns to work *with* the dynamics, not in isolation.

NeuralGCM (Google DeepMind / ECMWF, published in Nature 2024) demonstrated this approach at scale: a differentiable dynamical core in JAX ("Dinosaur") coupled with neural network parameterizations, trained on ERA5 reanalysis data, producing weather and climate forecasts competitive with the best existing methods. This is *exactly* the paradigm GT4Py + JAX could enable.

### 1.3 Sensitivity Analysis & Optimal Experimental Design

Adjoint models tell you: "If I change input X by a small amount, how much does output Y change?" This is used to:
- Determine which observations most constrain the forecast (observation targeting / optimal sensor placement)
- Attribute forecast errors to specific initial condition uncertainties
- Study ocean heat transport sensitivities (NASA ECCO project uses MITgcm's adjoint)

### 1.4 Parameter Estimation / Model Calibration

Climate models have many tunable parameters (e.g., drag coefficients, mixing rates). Rather than manual tuning, AD enables gradient-based optimization to find parameter values that best match observations. This is essentially a variant of 4D-Var where the control variable is model parameters instead of (or in addition to) initial conditions.

---

## 2. Forward Mode vs. Reverse Mode, Tangent-Linear vs. Adjoint, and the Taylor Test

### 2.1 The Two Modes of AD

Consider a function `F: ℝⁿ → ℝᵐ` with Jacobian `J = ∂F/∂x` (an m×n matrix).

**Forward mode (= tangent-linear model, TLM)**
- Computes **Jacobian-vector products (JVPs)**: `J · δx` for a given perturbation direction `δx ∈ ℝⁿ`
- One forward-mode pass gives you the derivative of *all m outputs* with respect to *one input direction*
- Cost: ~same as one forward model evaluation
- Efficient when **n is small** (few inputs, many outputs)
- In JAX: `jax.jvp(f, (x,), (dx,))`

**Reverse mode (= adjoint model)**
- Computes **vector-Jacobian products (VJPs)**: `Jᵀ · δy` for a given adjoint vector `δy ∈ ℝᵐ`
- One reverse-mode pass gives you the derivative of *one scalar output* (or one output direction) with respect to *all n inputs*
- Cost: ~2-4× one forward model evaluation (but independent of n!)
- Efficient when **m is small** (few outputs, many inputs)
- In JAX: `jax.grad(f)(x)` (for scalar output) or `jax.vjp(f, x)` (general)

### 2.2 Why This Terminology Exists in Weather/Climate

The terms "tangent-linear model" and "adjoint model" predate the ML/AD communities' use of "forward mode" and "reverse mode." They come from the variational data assimilation literature (Talagrand, Courtier, Le Dimet — 1980s-90s).

| AD community term | Weather/climate term | JAX function | What it computes |
|---|---|---|---|
| Forward mode | Tangent-linear model (TLM) | `jax.jvp` | `J · δx` (sensitivity propagated forward) |
| Reverse mode | Adjoint model | `jax.vjp` / `jax.grad` | `Jᵀ · δy` (sensitivity propagated backward) |

**In 4D-Var specifically:**
1. The **tangent-linear model** propagates small perturbations `δx₀` forward in time: how does a small change in initial conditions affect the forecast?
2. The **adjoint model** propagates the gradient of the cost function `∇J` backward in time: how should we adjust the initial conditions to reduce the cost?

For data assimilation, the cost function `J(x₀)` is a *scalar*, but `x₀` has millions of components → **reverse mode (adjoint)** is overwhelmingly preferred (one backward pass gives the full gradient, vs. millions of forward passes).

For sensitivity studies where you perturb a single parameter and want to see the effect on all model fields, forward mode is natural.

### 2.3 The Taylor Test

The Taylor test is the **standard way to verify that your AD-computed gradient is correct**. It is based on Taylor's theorem. For a differentiable functional `J(m)` with gradient `∇J`:

**First-order remainder (no gradient information):**
```
|J(m + hδm) - J(m)| = O(h)    as h → 0
```
This converges to zero at rate h (order 1). If you halve h, the remainder halves.

**Second-order remainder (using the gradient):**
```
|J(m + hδm) - J(m) - h·∇J·δm| = O(h²)    as h → 0
```
This converges to zero at rate h² (order 2). If you halve h, the remainder *quarters*.

**How to run the test:**
1. Choose a perturbation direction `δm` (often random)
2. For a sequence of decreasing `h` values (e.g., h, h/2, h/4, h/8, ...):
   - Compute the first-order remainder: `r₁ = |J(m + hδm) - J(m)|`
   - Compute the second-order remainder: `r₂ = |J(m + hδm) - J(m) - h·∇J·δm|`
3. Check convergence rates:
   - `r₁` should decrease by factor ~2 each time (rate ~1) — this just confirms J is Lipschitz, sanity check
   - `r₂` should decrease by factor ~4 each time (rate ~2) — **this confirms the gradient is correct**

If the second-order remainder converges at rate 2, your adjoint/gradient is verified. If it doesn't, something is wrong — a bug, a non-differentiable operation, or an incorrect adjoint.

**In pseudocode (JAX):**
```python
def taylor_test(J_func, grad_J, m, dm, h_init=1e-3, n_steps=5):
    """Verify gradient by checking second-order convergence."""
    h = h_init
    J0 = J_func(m)
    dJdm = grad_J(m)  # or jax.grad(J_func)(m)
    directional_deriv = jnp.sum(dJdm * dm)

    for i in range(n_steps):
        J_pert = J_func(m + h * dm)
        r1 = abs(J_pert - J0)              # should be O(h)
        r2 = abs(J_pert - J0 - h * directional_deriv)  # should be O(h²)
        print(f"h={h:.1e}  |r1|={r1:.6e}  |r2|={r2:.6e}")
        h /= 2

    # Check: ratios of consecutive r2 should be ~4 (= 2²)
```

**Why it matters for GT4Py:** When you make a model differentiable through JAX, the Taylor test is the first thing you should run to verify correctness. It's a non-negotiable validation step.

---

## 3. The Shallow Water Model as a Demonstration Vehicle

### 3.1 Why the Shallow Water Equations Are an Excellent Choice

The shallow water equations (SWE) are the canonical "minimal geophysical fluid dynamics" model. They capture the essential wave dynamics (gravity waves, Rossby waves) that matter for weather/climate, while being simple enough to implement, understand, and run quickly.

**The equations** (2D, on an f-plane or β-plane):

```
∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0                          (mass conservation)
∂u/∂t + u·∂u/∂x + v·∂u/∂y - fv = -g·∂h/∂x + Fᵤ          (x-momentum)
∂v/∂t + u·∂v/∂x + v·∂v/∂y + fu = -g·∂h/∂y + Fᵥ          (y-momentum)
```

where h = fluid depth, (u,v) = velocity, f = Coriolis parameter, g = gravity, F = forcing/dissipation.

**Why SWE are perfect for this demo:**
- They're the "dynamical core" of NeuralGCM's Dinosaur (just the simpler version)
- They support interesting dynamics: geostrophic adjustment, Rossby waves, vortex interactions
- They are well-established in data assimilation literature (many papers use SWE for 4D-Var demos)
- They are small enough to differentiate without memory issues
- The NCAR/SWM code and the NOAA-GSL SENA-shallow-water already have GT4Py versions

### 3.2 Suitability of NCAR/SWM

The NCAR SWM is a good starting point. It's a finite-difference solver on a regular grid — straightforward to express in GT4Py stencils. The existing GT4Py port (from the NCAR hackathon / NOAA-GSL SENA-shallow-water repo) may need updating to work with `gt4py.next`, but the structure is right.

**Key consideration:** For AD with JAX, the model must be written as a *pure function* (no in-place mutations, no global state). JAX's functional paradigm requires that the time-stepping function takes the current state and returns the next state, with no side effects. This may require some refactoring of the GT4Py version but is quite natural in `gt4py.next`.

---

## 4. Proposed Demonstration Examples

I recommend a progression of three examples, from simple to impressive, all based on the same shallow water model. Each builds on the previous one and demonstrates a different AD application.

### Example A: Gradient Verification with the Taylor Test (Foundation)

**Goal:** Verify that AD through the GT4Py shallow water model works correctly.

**Setup:**
1. Implement the SWE in GT4Py backed by JAX arrays
2. Define a scalar cost function, e.g., `J(x₀) = ½ ∫(h(x, T) - h_target(x))² dx` — the squared difference between the final height field and some target, integrated over the domain
3. Compute `∇J` with respect to the initial height field `h₀` using `jax.grad`
4. Run the Taylor test to verify second-order convergence

**What it demonstrates:** The model is truly differentiable end-to-end, and JAX's AD produces correct gradients through GT4Py stencils.

**Visualization:** A log-log plot of the Taylor remainders vs. h, showing slopes of 1 (first-order) and 2 (second-order).

**Estimated complexity:** Low. Mostly plumbing to connect GT4Py's JAX backend to `jax.grad`.

### Example B: 4D-Var Data Assimilation (The Killer Demo)

**Goal:** Recover the initial conditions of a shallow water flow from sparse, noisy observations at later times.

**Scenario — "twin experiment":**
1. **Truth run:** Start from a known initial condition (e.g., a Gaussian height perturbation that will generate gravity waves), run the model forward, and save the solution at several time steps
2. **Observations:** Sample the truth at a sparse set of grid points (e.g., 5-10% of the domain) at several time steps, and add Gaussian noise
3. **Background:** Start from a different (wrong) initial condition — e.g., flat or uniformly perturbed
4. **4D-Var optimization:** Minimize the cost function `J(x₀)` using L-BFGS (from `scipy.optimize.minimize` or `jax.scipy`), where gradients are computed by `jax.grad` through the time integration

**What it demonstrates:**
- Starting from a wrong initial condition, 4D-Var + AD recovers the true initial state
- The gradient from AD guides the optimization efficiently — show cost function decreasing over iterations
- Compare with finite-difference gradient: AD is *orders of magnitude faster* for high-dimensional state spaces

**Visualization:**
- Panel 1: True initial condition (Gaussian bump)
- Panel 2: Background (wrong) initial condition
- Panel 3: Recovered initial condition after 4D-Var — should closely match truth
- Panel 4: Cost function vs. iteration number
- Optional: Animation of the truth run, observations, and analysis run

**Estimated complexity:** Medium. The SWE model must be wrapped as a JAX-differentiable function. The 4D-Var loop itself is ~50 lines of code.

### Example C: Learning a Subgrid Parameterization (The NeuralGCM Analogy)

**Goal:** Train a small neural network to act as a forcing/dissipation term in the shallow water model, mimicking what NeuralGCM does at full scale.

**Scenario:**
1. **High-resolution "truth":** Run the SWE at high resolution (e.g., 256×256) with some forcing that generates turbulent-like dynamics
2. **Coarse model:** Run at low resolution (e.g., 64×64) — this will have errors because it can't resolve small-scale features
3. **Hybrid model:** Add a small neural network (e.g., a 3-layer CNN) that takes the coarse state as input and outputs a correction/forcing term. The entire system (coarse SWE solver + NN) is differentiable
4. **End-to-end training:** Unroll the hybrid model for several time steps, compare with coarsened high-res truth, and backpropagate through the combined physics+NN system to train the NN weights

**What it demonstrates:**
- This is the NeuralGCM paradigm in miniature
- The NN learns to compensate for the coarse model's errors
- "Online" training through the differentiable solver is stable; "offline" training (NN trained on snapshots) tends to cause drift when coupled back in

**Visualization:**
- Coarse model without NN: drifts/is inaccurate after many steps
- Coarse model + trained NN: closely tracks the high-res truth
- Optional: show what happens with offline-trained NN vs. online-trained NN

**Estimated complexity:** High. Requires integrating a JAX neural network (e.g., via Flax or Equinox) with the GT4Py model. But this is the most impressive demo and directly shows why differentiable physics matters.

---

## 5. Recommended Project Plan

### Phase 1: Foundation (2-3 weeks)

1. **Get GT4Py + JAX working:** Ensure `gt4py.next` stencils can execute with JAX arrays. Apply any needed patches.
2. **Implement SWE in GT4Py/JAX:** Start from the NCAR/SWM GT4Py port or the NOAA-GSL SENA-shallow-water version. Refactor to be a pure JAX-compatible function: `state_new = step(state, params, dt)`.
3. **Verify forward model:** Run a standard test case (e.g., geostrophic adjustment, Williamson test case 2) and confirm the model produces correct dynamics.

### Phase 2: AD Verification (1-2 weeks)

4. **Define a scalar cost function** on the final state.
5. **Compute gradient with `jax.grad`** through the time-stepping loop.
6. **Run the Taylor test** (Example A above). Debug until you get clean second-order convergence. This is the most important milestone — everything else builds on it.

### Phase 3: Data Assimilation Demo (2-3 weeks)

7. **Set up twin experiment** infrastructure: truth run, observation sampling, background state.
8. **Implement 4D-Var** optimization loop using `scipy.optimize.minimize(method='L-BFGS-B')` with JAX gradients.
9. **Create visualizations** showing the recovery of initial conditions.
10. **Write up** the example with explanations accessible to the GT4Py community.

### Phase 4 (Optional / Ambitious): Hybrid Model Demo (3-4 weeks)

11. **Set up high-res / low-res truth/model pair.**
12. **Define a simple NN parameterization** (e.g., using Equinox or Flax).
13. **Implement end-to-end training loop** through the differentiable SWE solver.
14. **Compare online vs. offline training.**

---

## 6. Technical Notes for Implementation

### 6.1 JAX Compatibility Requirements

For a GT4Py program to be differentiable with JAX:
- **No in-place mutation:** JAX arrays are immutable. Operations like `a[i] = b` must become `a = a.at[i].set(b)`. GT4Py stencils should naturally avoid this if they produce new output fields.
- **No Python control flow depending on array values:** `if array[i] > 0` breaks tracing. Use `jnp.where` instead. This is relevant for limiters, boundary conditions, etc.
- **Pure functions:** The time step should be a pure function `f(state) → state`. Use `jax.lax.scan` or `jax.lax.fori_loop` for the time loop instead of a Python for-loop (for efficiency under `jax.grad`).
- **Checkpointing:** For long time integrations, memory can blow up because reverse-mode AD stores intermediate states. Use `jax.checkpoint` (= `jax.remat`) to trade memory for recomputation.

### 6.2 JAX AD Cheat Sheet

```python
import jax
import jax.numpy as jnp

# Gradient of scalar function
grad_f = jax.grad(f)              # df/dx, where f: ℝⁿ → ℝ
grad_val = grad_f(x)

# Forward-mode (tangent-linear)
primals, tangents = jax.jvp(f, (x,), (dx,))   # f(x) and J·dx

# Reverse-mode (adjoint)
primals, vjp_fn = jax.vjp(f, x)
grad_x = vjp_fn(dy)               # Jᵀ·dy

# Gradient + value simultaneously
val, grad_val = jax.value_and_grad(f)(x)

# Efficient time loop for AD
def scan_step(state, _):
    return time_step(state), None  # carry, output

final_state, _ = jax.lax.scan(scan_step, initial_state, None, length=n_steps)

# Checkpointing for memory efficiency
@jax.checkpoint
def time_step(state):
    ...
```

### 6.3 Common Pitfalls

- **NaN gradients from `sqrt(0)` or `log(0)`:** Add small epsilon values.
- **Non-differentiable boundary conditions:** Periodic BCs are fine. Reflective BCs with if-else logic need `jnp.where`.
- **Very long time integrations:** Gradients can explode or vanish (the "chaotic adjoint" problem). For the demo, keep integration windows short (tens of time steps, not thousands).
- **Double precision:** Use `jax.config.update("jax_enable_x64", True)` — important for the Taylor test to see clean second-order convergence before roundoff dominates.

---

## 7. Summary: Why This Is Cool

The pitch for GT4Py + JAX + AD in one paragraph:

> *Today, making a weather/climate model differentiable requires either years of manual adjoint coding (as ECMWF does in Fortran) or rewriting the model from scratch in JAX (as NeuralGCM did). GT4Py offers a third path: write your model once in a high-level DSL, and get both high-performance execution on GPUs/CPUs **and** automatic differentiation for free via the JAX backend. This enables data assimilation, sensitivity analysis, and hybrid AI-physics models without any additional development effort for the adjoint.*

The shallow water demo makes this concrete and tangible: a small, self-contained model that runs a real 4D-Var data assimilation cycle using nothing but `jax.grad` through GT4Py stencils.

---

## References & Further Reading

- **NeuralGCM:** Kochkov et al. (2024), "Neural general circulation models for weather and climate," *Nature*. [doi:10.1038/s41586-024-07744-y](https://www.nature.com/articles/s41586-024-07744-y)
- **Dinosaur (differentiable dycore):** [github.com/neuralgcm/dinosaur](https://github.com/neuralgcm/dinosaur)
- **MITgcm/ECCO adjoint:** [ecco-group.org/adjoint.htm](https://ecco-group.org/adjoint.htm)
- **MITgcm-AD v2 with Tapenade:** Berdahl-Baldwin et al. (2024), *Future Generation Computer Systems*. [doi:10.1016/j.future.2024.09.011](https://www.sciencedirect.com/science/article/pii/S0167739X2400476X)
- **Taylor test (dolfin-adjoint docs):** [dolfin-adjoint.org/en/latest/documentation/verification.html](http://www.dolfin-adjoint.org/en/latest/documentation/verification.html)
- **JCM (JAX Circulation Model):** Davenport et al. (2026), "JCM v1.0: A Differentiable, Intermediate-Complexity Atmospheric Model," *EGUsphere preprint*.
- **NN tangent-linear/adjoint for DA:** Hatfield et al. (2021), "Building Tangent-Linear and Adjoint Models for Data Assimilation with Neural Networks," *JAMES*. [doi:10.1029/2021MS002521](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002521)
- **4D-Var with SWE:** Pires et al. (2017), "A 4D-Var Method with Flow-Dependent Background Covariances for the Shallow-Water Equations." [arXiv:1710.11529](https://arxiv.org/abs/1710.11529)
- **NOAA-GSL SENA-shallow-water (GT4Py port):** [github.com/NOAA-GSL/SENA-shallow-water](https://github.com/NOAA-GSL/SENA-shallow-water)
- **Tesseract 4D-Var tutorial (JAX + Lorenz 96):** [docs.pasteurlabs.ai/.../data-assimilation-4dvar.html](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/demo/data-assimilation-4dvar.html)
