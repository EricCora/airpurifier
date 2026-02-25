import numpy as np

from sim.core import AirRoomSimulation, PurifierConfig, RoomConfig, SourceConfig


def run_case(name, purifier_xy, flow=0.0665, steps=240, dt_req=0.05):
    sim = AirRoomSimulation(
        RoomConfig(), PurifierConfig(x=purifier_xy[0], y=purifier_xy[1], flow_m3s=flow), SourceConfig()
    )

    t = 0.0
    masses = []
    for _ in range(steps):
        dt = sim.step(dt_req)
        t += dt
        masses.append(sim.diagnostics(dt_req)["total_pollutant_mass"])

    diag = sim.diagnostics(dt_req)
    room_vol = sim.room.lx * sim.room.ly * sim.room.lz
    mean_c = float(np.mean(sim.c[~sim.solid]))

    k_mix = sim.purifier.filtration_efficiency * sim.purifier.flow_m3s / room_vol
    c_ss = sim.source.emission_rate / max(sim.purifier.filtration_efficiency * sim.purifier.flow_m3s, 1e-9)
    c_pred = c_ss * (1 - np.exp(-k_mix * t))

    dm_dt = (masses[-1] - masses[-2]) / sim.last_dt if len(masses) > 1 else 0.0
    mass_balance_residual = dm_dt - sim.source.emission_rate + diag["sink_mass_rate"]

    return {
        "case": name,
        "time": t,
        "mean_c": mean_c,
        "well_mixed_c": c_pred,
        "ratio_vs_wellmixed": mean_c / max(c_pred, 1e-12),
        "max_div": diag["max_divergence"],
        "mean_div": diag["mean_divergence"],
        "max_vel": diag["max_velocity"],
        "cfl": diag["cfl"],
        "dm_dt": dm_dt,
        "mass_balance_residual": mass_balance_residual,
    }


def main():
    cases = [
        ("center", (3.6, 2.6)),
        ("far_corner", (4.4, 3.4)),
        ("near_source", (1.6, 3.0)),
    ]
    rows = [run_case(name, xy) for name, xy in cases]

    print("case,time_s,mean_c,wellmixed_c,ratio,max_div,mean_div,max_vel,cfl,dm_dt,mass_balance_residual")
    for r in rows:
        print(
            f"{r['case']},{r['time']:.2f},{r['mean_c']:.3e},{r['well_mixed_c']:.3e},{r['ratio_vs_wellmixed']:.3f},"
            f"{r['max_div']:.3e},{r['mean_div']:.3e},{r['max_vel']:.3f},{r['cfl']:.3f},"
            f"{r['dm_dt']:.3e},{r['mass_balance_residual']:.3e}"
        )


if __name__ == "__main__":
    main()
