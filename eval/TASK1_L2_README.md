# Task 1 — L=2 direct-step diagnostics

Runner: `eval/eval_l2_stability_task1.py`.

The network stays the trained direct-step map `ema_vit(u, [z], t=1) -> (g, h)`.
Nothing is rewritten as Euler. Difference-quotient names (`F_u`, `H_eff`, `Sigma`)
are `(block - I) / dt` labels so they can sit next to the step-size deck.
`dt` defaults to 0.25 (five snapshots at 0.05).

Jacobian actions are centered finite differences. `torch.autograd.functional.jvp`
is not used. `D` is `avg_pool2d` with the checkpoint's `ds_factors[0]` (8 for
the 256→32 L=2 run). Fields inside the network are `Omega / --norm`. Reported
error norms are multiplied back by `--norm`.

Finite-difference step sizes are block-aware. One scalar `h` makes the larger
of the fine and latent relative perturbations equal `--fd-eps`. Each block is
scaled by `max(||block||, 1e-8 * sqrt(numel))`. A zero direction returns zeros.
The selected `--fd-eps` is compared with `eps/3` and `eps/10` on fine-only,
latent-only, and graph directions, and on `g` and `h` separately. Random
directions use `--seed`. That sweep is not sufficient. `model_fd_validated`
also requires `composition_rel_max` and full-map `fd_additivity_rel` to be
within `--fd-tol`, and the mixed retention and linearized directions at an
early, middle, and late step. If the initial epsilon check fails and
`--allow-unvalidated-fd` is absent, the run writes metadata and exits 2
before the diagnostic windows. `--spectral-radius` runs Arnoldi only after
`S` itself passes the same tolerance; `spectral_validated` stays false
otherwise. `--physical-jvp py2d` records a separate
`physical_tangent_validated` from the Task 1 frames at their own `nx` and
`--dt`. Proxy mode stores `not_applicable` there, not true.

## Metric → equation

`DB`, `D Phi'_n B`, and `D Phi'_{n+1} B` are different operators.

- `tr_A_*`, `tr_A_over_N`: Hutchinson trace of `A = dg/du`. Finite-difference estimate.
- `tr_K_*`: trace of `K = dh/dz`. Finite-difference estimate.
- `tr_DB_*`, `tr_DB_over_N`: coarse-probe trace of `D B`. This is the difference-quotient quantity behind `F_z D`.
- `tr_DFz_over_N`: `Tr(DB) / (N dt)`. Difference-quotient label, including in py2d mode.
- `tr_B_DPhi_n_*`: fine-probe trace of `B D Phi'_n`. In proxy mode `D Phi' ≈ D`, so this estimates `Tr(BD) = Tr(DB)`. In py2d mode it is the physical feedback trace and is not `Tr(DB)`.
- `tr_DPhi_next_B_*`: coarse-probe trace of `D Phi'_{n+1} B`, used only inside `S`.
- `composition_rel_*`: `||Jv - Av - B(w)|| / max(||Jv||, floor)` from three separate finite differences. Not `Jv - Av` reused as `B(w)`.
- `trace_estimator_gap`: fine-probe mean of `v·B(w)` minus coarse-probe mean of `w·D(Bw)`, with `trace_estimator_gap_ci95`. In proxy mode both estimate `Tr(BD)`. This is sampling uncertainty, not the per-vector composition residual.
- `tr_J_*`, `tr_J_over_N`: trace of `J = A + B D Phi'_n` (`D Phi' ≈ D` in proxy mode).
- `tr_Fu_over_N`: `(Tr(A) - N) / (N dt)`. Its error bar is `tr_A_ci95 / (N dt)`.
- `tr_Heff_over_N`: `(Tr(J) - N) / (N dt)`. Difference-quotient label.
- `tr_S_*`, `tr_S_over_M`: trace of `S = K - D Phi'_{n+1} B`.
- `tr_Sigma_over_M`: `(Tr(S) - M) / (M dt)`. Difference-quotient label.
- `amp_J_*`: mean of `||Jv||^2 / N` over Rademacher probes. Mean-square directional amplification, not contraction in every direction.
- `offgraph_probe_*`: `||L^off v||^2 / N`.
- `fd_additivity_rel`: `||FD(v+w) - FD(v) - FD(w)|| / max(||FD(v+w)||, floor)` on the full network. Above `--fd-tol` it fails `model_fd_validated`.
- `s_additivity_rel`: the same residual for the `S` matvec used by Arnoldi. Above `--fd-tol`, Arnoldi is not run and `spectral_validated` is false.
- `rho_S`: leading magnitude of `S`, only after `S` additivity passes. Optional `--spectral-radius`.
- `roll_eps_u_norm`, `roll_eps_z_norm`: teacher-forced one-step residuals.
- `roll_e_norm`, `roll_eta_norm`: free-rollout fine and latent errors from `(u_0, D u_1)`.
- `roll_I`, `roll_R`: `I_n = ||sum_{i<n} eps_u||`, `R_n = ||e_n|| - I_n`.
- `roll_rho_norm`: `||e_n - sum_{i<n} eps_u||`. Not the same as `R_n`.
- `roll_rH_norm`, `roll_rZ_norm`: truncated first-factor channels. A large residual is not automatically a bug.
- `roll_recon_norm`: `||rho - r_H - r_z||`.
- `roll_cos_rH_rZ`: alignment. Undefined when either channel is zero; plotted on its own panel with limits `[-1, 1]`.
- `roll_e_lin_norm`, `roll_eta_lin_norm`: full linearized errors. No physical tangent.
- `roll_e_lin_err`, `roll_eta_lin_err`: `||e - e_lin||` and `||eta - eta_lin||`, vector differences.
- `roll_recon_lin_norm`: `||rho_lin - r_H - r_z||`.

`approximation` is `proxy` or `py2d_fd`. Neither label means the finite difference is an exact derivative. Proxy means `D Phi' ≈ D`.

## Py2D

`--physical-jvp py2d` builds the tangent with `Py2DInMemory` in
`eval/py2d_inmem.py`, which imports kernels from `/global/homes/d/divyam/py2d`
(`gridgen`, `initialize_wavenumbers_rfft2`, `Omega2Psi_spectral`,
`Psi2UV_spectral`, `convection_conserved_dealias`). The step is the NoSGS
AB2/CN update (Re=1e4, fkx=fky=4, alpha=0.1, beta=20, solver dt=1e-4).
`Py2D_solver` itself is not called. Jax is imported only when that tangent
is constructed. The `hine` env is not modified. Use
`nersc_job_launchers/make_hine_jax_env.sh`. The grid size follows the field
(256 for this L=2 checkpoint).

## L=3

Not implemented. L=3 needs two off-graph errors and a parasitic block of
size `M1+M2`. The note (assumption A4) says the `z^{(2)}` index convention
still has to be checked. The finite-difference pair can take a list of
coarse fields; the operator algebra and the retention sums cannot be copied
from L=2.

## Smoke

```bash
sbatch nersc_job_launchers/eval_l2_task1_debug.sbatch
```

Results go to `.../plus_L2_m32/task1_l2_debug_test/`, not the production
Jacobian directories.
