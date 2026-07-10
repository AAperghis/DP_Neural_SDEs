from pathlib import Path
import numpy as np
from scipy.io import loadmat

if __name__ == "__main__":
    qtf_path = Path("src/supply.mat")
    with open(qtf_path, "rb") as f:
        qtf_data = loadmat(f)

    # --- Drift force coefficients ---
    drift_coeffs = np.stack(
        qtf_data["vessel"]["driftfrc"][0][0][0]["amp"][0][0].flatten()
    )[:3, :, :, 0]
    empty_coeffs = np.zeros((36, 36))
    drift_coeffs = np.stack(
        [
            drift_coeffs[0],
            drift_coeffs[1],
            empty_coeffs,
            empty_coeffs,
            empty_coeffs,
            drift_coeffs[2],
        ]
    )  # DOF with zeros
    freqs = qtf_data["vessel"]["driftfrc"][0][0][0]["w"][0][0].flatten()
    headings = qtf_data["vessel"]["headings"][0][0][0].flatten()

    out_path = Path("src/thesis/full_order_dp/drift_coefficients.npz")
    np.savez(
        out_path,
        drift_coefficients=drift_coeffs,
        frequencies=freqs,
        headings=headings,
    )
    print(f"Saved drift coefficients to {out_path}")
    print(f"  drift_coefficients: {drift_coeffs.shape}  (dof, freq, heading)")
    print(f"  frequencies:        {freqs.shape}")
    print(f"  Frequency range:    {freqs.min()} - {freqs.max()}")
    print(f"  headings:           {headings.shape}")

    # --- Force RAOs (first-order) ---
    rao = qtf_data["vessel"]["forceRAO"][0][0][0]
    rao_amp_raw = rao["amp"][0][0]  # (6,) object array
    rao_phase_raw = rao["phase"][0][0]  # (6,) object array
    rao_w = rao["w"][0][0].flatten()

    # Each DOF is (freq, heading, vel) — extract vel=0 only
    rao_amp = np.stack(
        [rao_amp_raw[dof][:, :, 0] for dof in range(6)]
    )  # (6, freq, heading)
    rao_phase = np.stack(
        [rao_phase_raw[dof][:, :, 0] for dof in range(6)]
    )  # (6, freq, heading)

    rao_path = Path("src/thesis/full_order_dp/force_rao.npz")
    np.savez(
        rao_path,
        amplitude=rao_amp,
        phase=rao_phase,
        frequencies=rao_w,
        headings=headings,
    )
    print(f"\nSaved force RAOs to {rao_path}")
    print(f"  amplitude: {rao_amp.shape}  (dof, freq, heading)")
    print(f"  phase:     {rao_phase.shape}  (dof, freq, heading)")
    print(f"  frequencies: {rao_w.shape}")
    print(f"  headings:    {headings.shape}")
