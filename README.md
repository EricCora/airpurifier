# Room-Scale CFD + Pollutant Transport Simulator (Levoit Core 300 centered)

## 1) Levoit Core 300 research summary (known vs assumed)

I attempted direct retrieval of official/manual pages from this runtime, but requests returned `HTTP 403`, so values below are explicitly flagged as **externally verifiable assumptions**.

Reference links (verify outside this environment):
- Official product page: <https://levoit.com/products/core-300-air-purifier>
- Official owner/manual page: <https://levoit.com/pages/core-300-owners>
- Manual mirror: <https://manuals.plus/levoit/core-300-true-hepa-air-purifier-manual>

Parameters used:
- Device geometry: ~8.7 in diameter x 14.2 in height ≈ 0.221 m x 0.360 m.
- CADR: ~141 CFM.
- Intake/discharge approximation: 360° side intake + top upward outlet.
- Mode mapping used in UI:
  - Sleep ≈ 0.024 m³/s,
  - Medium ≈ 0.050 m³/s,
  - High ≈ 0.069 m³/s.

Unit conversion:
- 141 CFM × 0.000471947 = **0.0665 m³/s**.

## 2) Mathematical model

### Airflow (CFD-lite fallback)
Because interactive, high-resolution LES/RANS is not feasible here, the solver uses a coarse-grid incompressible projection method:

\[
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}
= -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u} + \mathbf{f}_{purifier},
\qquad \nabla\cdot\mathbf{u}=0
\]

Numerics:
- semi-Lagrangian advection (more robust for interactive timesteps),
- explicit diffusion,
- pressure Poisson solve (Jacobi iterations),
- projection correction,
- adaptive timestep from CFL + diffusion constraints.

### Pollutant transport
\[
\frac{\partial c}{\partial t} + \mathbf{u}\cdot\nabla c = D\nabla^2c + S_{dog}(\mathbf{x}) - k_{purifier}(\mathbf{x}) c
\]

- Dog bed source term \(S_{dog}\): user-controlled emission (kg/s proxy) distributed over source cells.
- Purifier sink term: local first-order decay in intake shell with
\[
k = \eta Q / V_{shell}
\]
where \(\eta\)=filtration efficiency and \(Q\)=flow rate.

### Boundary conditions and solids
- No-slip walls: floor, ceiling, and room walls.
- Furniture represented as solid block obstacles with no-slip treatment.

## 3) Purifier room-scale model

Purifier is represented as both:
1. **Momentum/flow inducer**
   - Upward top discharge region with velocity tied to \(Q/A_{outlet}\).
   - Radial inward side shell forcing to mimic 360° intake.
2. **Pollutant sink**
   - Concentration in intake shell decays at rate derived from throughput \(\eta Q\).

This is a room-scale surrogate, not a blade-/filter-resolved internal CFD model.

## 4) Interactive program

`app.py` provides controls for:
- purifier position,
- fan mode / flow override,
- dog-bed emission rate,
- requested timestep and substeps,
- run / pause / reset / single-step.

Visual outputs:
- pollutant concentration heatmap on selectable z-slice,
- 3D airflow vectors,
- diagnostics: dt-used, CFL, divergence residuals, max velocity,
- source/sink mass-rate monitoring,
- well-mixed baseline comparison (ACH, predicted C(t), simulated room-average C).

## 5) Validation and trust limits

Included checks:
- incompressibility residual (`max|div u|`, `mean|div u|`),
- CFL + adaptive dt,
- mass-rate diagnostics (`source_rate`, `sink_rate`),
- scenario sensitivity (`center`, `corner`, `near_source`),
- comparison to well-mixed baseline trajectory.

### What is trustworthy
- Relative placement trends (qualitative): center/corner/near-source comparisons.
- First-order interaction between advection, diffusion, and purifier removal.
- Stability-aware transient behavior via semi-Lagrangian + projection.

### Remaining approximations
- Coarse grid; near-wall/turbulence details are not resolved.
- No explicit RANS/LES turbulence closure.
- No thermal buoyancy, deposition, humidity, or multi-size particle classes.
- Core 300 specs here should be verified against official docs for quantitative decisions.

## 6) Usage

Install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run app:
```bash
streamlit run app.py
```

Run validation sweep:
```bash
python -m sim.validation
```

## 7) Next iteration (recommended)

1. OpenFOAM/FEniCSx transient solver with finer mesh.
2. Turbulence closure (k-omega SST or LES SGS).
3. Empirical purifier velocity profile + pressure-flow curve.
4. Size-resolved particles (PM0.3/1/2.5/10) with deposition/settling.
5. Thermal buoyancy and HVAC supply/return coupling.
6. Calibration with measured PM sensor data.
