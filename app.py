import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sim.core import AirRoomSimulation, PurifierConfig, RoomConfig, SourceConfig

st.set_page_config(page_title="Room CFD + Pollutant Simulator", layout="wide")
st.title("Levoit Core 300 Room Airflow + Pollutant Simulation")
st.caption("Fidelity: coarse 3D projection CFD (semi-Lagrangian advection + pressure projection).")

if "sim" not in st.session_state:
    st.session_state.sim = AirRoomSimulation(RoomConfig(), PurifierConfig(), SourceConfig())
    st.session_state.time = 0.0
    st.session_state.running = False

sim: AirRoomSimulation = st.session_state.sim

with st.sidebar:
    st.header("Controls")
    px = st.slider("Purifier X (m)", 0.3, sim.room.lx - 0.3, float(sim.purifier.x), 0.1)
    py = st.slider("Purifier Y (m)", 0.3, sim.room.ly - 0.3, float(sim.purifier.y), 0.1)
    mode = st.selectbox("Levoit mode", ["Sleep", "Medium", "High"], index=1)
    mode_flow = {"Sleep": 0.024, "Medium": 0.050, "High": 0.069}
    flow = st.slider("Flow override (m³/s)", 0.015, 0.090, mode_flow[mode], 0.001)

    emission = st.slider(
        "Dog bed emission (kg/s proxy)",
        1e-8,
        1e-5,
        float(sim.source.emission_rate),
        1e-8,
        format="%.1e",
    )

    dt_request = st.slider("Requested timestep (s)", 0.01, 0.20, 0.05, 0.005)
    substeps = st.slider("Substeps/frame", 1, 30, 5)
    zslice = st.slider("Concentration Z-slice (m)", 0.1, sim.room.lz - 0.1, 0.9, 0.05)

    c1, c2, c3 = st.columns(3)
    if c1.button("Run"):
        st.session_state.running = True
    if c2.button("Pause"):
        st.session_state.running = False
    if c3.button("Reset"):
        sim.reset()
        st.session_state.time = 0.0

sim.set_purifier(px, py, flow)
sim.set_source_emission(emission)

if st.session_state.running:
    for _ in range(substeps):
        dt_used = sim.step(dt_request)
        st.session_state.time += dt_used

if st.button("Step once"):
    dt_used = sim.step(dt_request)
    st.session_state.time += dt_used

diag = sim.diagnostics(dt_request)
x = (np.arange(sim.room.nx) + 0.5) * sim.dx
y = (np.arange(sim.room.ny) + 0.5) * sim.dy
z = (np.arange(sim.room.nz) + 0.5) * sim.dz
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

st.subheader(f"Simulation time: {st.session_state.time:.2f} s")
r1, r2, r3, r4 = st.columns(4)
r1.metric("dt used", f"{diag['dt_used']:.3f} s")
r2.metric("CFL", f"{diag['cfl']:.2f}")
r3.metric("Max |div u|", f"{diag['max_divergence']:.2e} 1/s")
r4.metric("Mean |div u|", f"{diag['mean_divergence']:.2e} 1/s")

r5, r6, r7, r8 = st.columns(4)
r5.metric("Max velocity", f"{diag['max_velocity']:.2f} m/s")
r6.metric("Pollutant mass", f"{diag['total_pollutant_mass']:.2e} kg")
r7.metric("Source rate", f"{diag['source_mass_rate']:.2e} kg/s")
r8.metric("Filtration sink", f"{diag['sink_mass_rate']:.2e} kg/s")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Pollutant concentration slice")
    kz = int(np.argmin(np.abs(z - zslice)))
    fig_c = go.Figure(
        data=go.Heatmap(
            x=x,
            y=y,
            z=sim.c[:, :, kz].T,
            colorscale="Turbo",
            colorbar=dict(title="kg/m³"),
        )
    )
    fig_c.update_layout(height=480, xaxis_title="x (m)", yaxis_title="y (m)")
    st.plotly_chart(fig_c, width="stretch")

with col2:
    st.markdown("#### Airflow vectors")
    sx, sy, sz = 3, 3, 3
    mask = ~(sim.solid[::sx, ::sy, ::sz].flatten())
    fig_u = go.Figure(
        data=go.Cone(
            x=X[::sx, ::sy, ::sz].flatten()[mask],
            y=Y[::sx, ::sy, ::sz].flatten()[mask],
            z=Z[::sx, ::sy, ::sz].flatten()[mask],
            u=sim.u[::sx, ::sy, ::sz].flatten()[mask],
            v=sim.v[::sx, ::sy, ::sz].flatten()[mask],
            w=sim.w[::sx, ::sy, ::sz].flatten()[mask],
            colorscale="Blues",
            sizemode="scaled",
            sizeref=1.3,
            showscale=False,
        )
    )
    fig_u.update_layout(
        height=480,
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            xaxis=dict(range=[0, sim.room.lx]),
            yaxis=dict(range=[0, sim.room.ly]),
            zaxis=dict(range=[0, sim.room.lz]),
        ),
    )
    st.plotly_chart(fig_u, width="stretch")

st.markdown("### Well-mixed baseline (for sanity-check)")
room_vol = sim.room.lx * sim.room.ly * sim.room.lz
k_mix = sim.purifier.filtration_efficiency * sim.purifier.flow_m3s / room_vol
c_ss = sim.source.emission_rate / max(sim.purifier.filtration_efficiency * sim.purifier.flow_m3s, 1e-9)
c_pred = c_ss * (1.0 - np.exp(-k_mix * st.session_state.time))
mean_c = float(np.mean(sim.c[~sim.solid]))

b1, b2, b3 = st.columns(3)
b1.metric("ACH (purifier only)", f"{sim.purifier.flow_m3s * 3600 / room_vol:.2f} h⁻¹")
b2.metric("Well-mixed C(t)", f"{c_pred:.2e} kg/m³")
b3.metric("Sim room-avg C", f"{mean_c:.2e} kg/m³")
