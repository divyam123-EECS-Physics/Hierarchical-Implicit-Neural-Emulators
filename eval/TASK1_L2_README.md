# Task 1 — L=2 direct-step diagnostics

Runner: `eval/eval_l2_stability_task1.py`.

The network stays the trained direct-step map `ema_vit(u, [z], t=1) -> (g, h)`.
Nothing is rewritten as Euler. Difference-quotient names (`F_u`, `H_eff`, `Sigma`)
are `(block - I) / dt` labels so they can sit next to the step-size deck.
`dt` defaults to 0.05, one kept frame. The runner checks the `time` value
stored in each eval mat and aborts if consecutive frames are not `--dt` apart.

Network blocks are `torch.autograd.functional.jvp` in float64. The physical
tangent is `jax.jvp` of the 512 DNS advance, embedded and spectrally truncated
back to the model grid. Centered differences are not the Task 1 operator. `D` is `avg_pool2d` with the checkpoint's `ds_factors[0]` (8 for
the 256→32 L=2 run). Fields inside the network are `Omega / --norm`. Reported
error norms are multiplied back by `--norm`.

`--production` selects the configuration used by the completed float64 run:
py2d, 20 windows, rollout 40, 32 probes, `dt=0.05`, spectral radius, and the
corrected sample index. The argparse defaults stay at the old development
values so a bare command is not silently relabeled as that run. A flag passed
on the command line overrides the preset.

JVP additivity and composition are linearity checks of the autodiff operator.
They are not an independent derivative validation. A dense
`torch.autograd.functional.jacobian` reference and a labeled centered
difference may be used in tests. Those values are not written as the reported
traces. If `jax.jvp` or `torch.autograd.functional.jvp` fails, the run fails.
`Py2DFDTangent` is not a fallback.

`--spectral-radius` uses `scipy.sparse.linalg.eigs` on a float64
`LinearOperator`. `plots/rho_S.png` is labeled validated only when the operator
gate and the eigenpair residual both pass. The residual does not prove that
the returned value is the dominant eigenvalue. Flow replay, JVP additivity,
and the eigenpair residual use separate tolerances (`1e-3`, `1e-6`, `1e-3`).
The physical map is the single-field startup: `Omega0` and `convec0` are
functions of the same 512 field, and no previous snapshot is held fixed.

## Metric → equation

`DB`, `D Phi'_n B`, and `D Phi'_{n+1} B` are different operators.

- `tr_A_*`, `tr_A_over_N`: Hutchinson trace of `A = dg/du`. Stochastic estimate from `torch.autograd.functional.jvp`, not an exact trace.
- `tr_K_*`: Hutchinson trace of `K = dh/dz`. Stochastic estimate, not an exact trace.
- `tr_DB_*`, `tr_DB_over_N`: coarse-probe trace of `D B`. This is the difference-quotient quantity behind `F_z D`.
- `tr_DFz_over_N`: `Tr(DB) / (N dt)`. Difference-quotient label, including in py2d mode.
- `tr_B_DPhi_n_*`: fine-probe trace of `B D Phi'_n`. In proxy mode `D Phi' ≈ D`, so this estimates `Tr(BD) = Tr(DB)`. In py2d mode it is the physical feedback trace and is not `Tr(DB)`.
- `tr_DPhi_next_B_*`: coarse-probe trace of `D Phi'_{n+1} B`, used only inside `S`.
- `composition_rel_*`: `||Jv - Av - B(w)|| / max(||Jv||, floor)` from three separate JVPs. Not `Jv - Av` reused as `B(w)`.
- `trace_estimator_gap`: fine-probe mean of `v·B(w)` minus coarse-probe mean of `w·D(Bw)`, with `trace_estimator_gap_ci95`. In proxy mode both estimate `Tr(BD)`. This is sampling uncertainty, not the per-vector composition residual.
- `tr_J_*`, `tr_J_over_N`: trace of `J = A + B D Phi'_n` (`D Phi' ≈ D` in proxy mode).
- `tr_Fu_over_N`: `(Tr(A) - N) / (N dt)`. Its error bar is `tr_A_ci95 / (N dt)`.
- `tr_Heff_over_N`: `(Tr(J) - N) / (N dt)`. Difference-quotient label.
- `tr_S_*`, `tr_S_over_M`: trace of `S = K - D Phi'_{n+1} B`.
- `tr_Sigma_over_M`: `(Tr(S) - M) / (M dt)`. Difference-quotient label.
- `amp_J_*`: mean of `||Jv||^2 / N` over Rademacher probes. Mean-square directional amplification, not contraction in every direction.
- `offgraph_probe_*`: `||L^off v||^2 / N`.
- `fd_additivity_rel`: `||J(v+w) - J(v) - J(w)|| / max(||J(v+w)||, floor)` on the full network. The key name is legacy. Above the JVP tolerance it fails `model_fd_validated`.
- `s_additivity_rel`: the same residual for the `S` matvec used by Arnoldi. Above `--fd-tol`, Arnoldi is not run and `spectral_validated` is false.
- `rho_S`: leading magnitude of `S`, only after `S` additivity passes. Optional `--spectral-radius`.
- `roll_eps_u_norm`, `roll_eps_z_norm`: teacher-forced one-step residuals.
- `roll_e_norm`, `roll_eta_norm`: free-rollout fine and latent errors from `(u_0, D u_1)`.
- `roll_I`, `roll_R`: `I_n = ||sum_{i<n} eps_u||`, `R_n = ||e_n|| - I_n`.
- `roll_rho_norm`: `||e_n - sum_{i<n} eps_u||`. Not the same as `R_n`.
- `roll_rH_norm`, `roll_rZ_norm`: truncated first-factor channels. A large residual is not automatically a bug.
- `roll_recon_norm`: `||rho - r_H - r_z||`.
- `roll_cos_eta_inj`: legacy name for `cos(eta_n, sum_{j<n} xi_j)`. It is not `sum epsilon^z`.
- `roll_cos_eta_eps_z`: `cos(eta_n, sum_{j<n} epsilon^z_j)`.
- `roll_cos_nu_xi`: `cos(nu_n, sum_{j<n} xi_j)`, with `nu_n = eta_n - D Phi'_n e_n`.
- `roll_eps_z_sum_norm`, `roll_xi_sum_norm`: norms of the accumulated vectors, not sums of norms.
- `roll_cos_*`: undefined when either vector is zero. Summary figures show the mean of defined per-window cosines and a 16th–84th percentile band, not per-window lines.
- `roll_e_lin_norm`, `roll_eta_lin_norm`: full linearized errors. No physical tangent.
- `roll_e_lin_err`, `roll_eta_lin_err`: `||e - e_lin||` and `||eta - eta_lin||`, vector differences.
- `roll_recon_lin_norm`: `||rho_lin - r_H - r_z||`.
- `roll_p_norm`: `p_n = C_n e_n`, the latent half of `jvp_both_u`.
- `roll_q_norm`: `q_n = (K_n - I) eta_n`.
- `roll_b_norm`: `b_n = K_n eta_n + epsilon^z_n`.
- `roll_remainder_norm`: `eta_{n+1} - (p_n + b_n)`.
- `roll_I_z_norm`: `||sum_{k<n} epsilon^z_k||`. This is the same sum as `roll_eps_z_sum_norm`; the older key is unchanged.
- `roll_r_u_to_z_norm`: `r_H^eta = sum_{k<n} C_k e_k = dt sum_{k<n} E_{u,k} e_k`. Old key name kept.
- `roll_r_z_to_z_norm`: `r_z^eta = sum_{k<n} (K_k - I) eta_k = dt sum_{k<n} E_{z,k} eta_k`. Old key name kept.
- `roll_rho_eta_norm`: `rho^eta = eta_n - sum_{k<n} epsilon^z_k`. This is not `R_n^eta`.
- `R_n^eta = ||eta_n|| - ||sum_{k<n} epsilon^z_k||`. It is a difference of norms, drawn from `roll_eta_norm` and `roll_I_z_norm`. It is not stored as its own key.
- `roll_channel_remainder_norm`: `||rho^eta - r_H^eta - r_z^eta||`, the Taylor remainder. On the linearized recursion with zero initial error this is zero. Older saved results do not contain this key.
- `roll_eta_recon_norm`: `||rho^eta - r_H^eta - r_z^eta - sum(remainder)||`. This closure also removes the accumulated one-step remainder. It is not the Taylor remainder above.
- `roll_cos_eta_rho`, `roll_cos_rz_eta`, `roll_cos_rho_rz`, `roll_cos_rho_Iz`: `cos(eta, rho^eta)`, `cos(eta, r_z^eta)`, `cos(rho^eta, r_z^eta)`, and `cos(rho^eta, sum epsilon^z)`. Older saved results do not contain these keys. They are not inferred from norms.
- `roll_r_u_to_z_first_norm`, `roll_r_z_to_z_first_norm`: truncated first-factor sums, labeled `(r_H^eta)^[1]` and `(r_z^eta)^[1]`. They apply `C` and `(K-I)` to accumulated injections. They are not `r_H^eta` or `r_z^eta`.
- `roll_cos_p_*`, `roll_cos_ru_*`, `roll_cos_Qe_eta`, `roll_cos_p_nu`: NaN when either vector is numerically zero. `Q = D Phi'`; proxy mode is `Q = D` and is labeled that way.
- `roll_delta_fine`: `2<b,p> + ||p||^2 = ||b+p||^2 - ||b||^2`. Negative means the fine-input term reduces the predicted next latent squared norm relative to omitting it, at this fixed linearization. It is not a nonlinear rollout ablation.
- `roll_lin_*`: the same decomposition on `e_lin`, `eta_lin`. Its reconstruction residual is numerical error. Omitted (NaN) when retention is off, because that path does not build `e_lin`.
- `tr_Ez_over_M`: `(Tr K - M) / (M dt)`.
- `tr_minus_Qnext_B_over_Mdt`: `-Tr(Q_{n+1} B) / (M dt)`, from the same coarse probes as `Tr K` and `Tr S`.
- `tr_S_split_probe_max`: largest probe-wise `|Tr K - Tr(Q_{n+1} B) - Tr S|`.
- `tr_C_Dplus_quot`: `Tr(C Dplus) / (M dt)`. `Dplus` copies each coarse value onto its pooling block, so `D Dplus = I_M`. The quotient is the fine-to-latent tendency under those lifted coarse perturbations. It is not `Tr(C)`, not `Tr(E_u)`, and not an autonomous latent contraction criterion.
- `gain_C_over_N`: `||C||_F^2 / N`, the mean of `||C v||^2 / N` on the fine Rademacher probes. `gain_Eu_sq_over_N` divides that by `dt^2` and is a squared gain, not signed damping.

`approximation` is `proxy` or `py2d_jvp`. Proxy means `D Phi' ≈ D`. `py2d_jvp` is `jax.jvp` of the 512 startup map, not a finite difference and not an identity. `roll_cos_eta_inj` remains `cos(eta_n, sum_{j<n} xi_j)`.

## One-window smoke, not submitted

```bash
python eval/eval_l2_stability_task1.py --production \
  --sample-index /pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task1_l2_n20_float64_py2d_dt05/sample_index.json \
  --n-snapshots 1 --n-gpus 1 \
  --checkpoint /pscratch/sd/d/divyam/hine_runs/plus_L2_m32/checkpoint-last.pth \
  --save-dir /pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task1_l2_latent_channels_smoke
```

## Py2D

`--physical-jvp py2d` builds the tangent with `Py2DInMemory` in
`eval/py2d_inmem.py`, which imports kernels from `/global/homes/d/divyam/py2d`
(`gridgen`, `initialize_wavenumbers_rfft2`, `Omega2Psi_spectral`,
`Psi2UV_spectral`, `convection_conserved_dealias`). The step is the NoSGS
AB2/CN update (Re=1e4, fkx=fky=4, alpha=0.1, beta=20, solver dt=1e-4).
`Py2D_solver` itself is not called. Jax is imported only when that tangent
is constructed. The `hine` env is not modified. Use
`nersc_job_launchers/make_hine_jax_env.sh`. The physical advance is the 512 DNS
field. The model state stays on the 256 grid and meets that field by spectral
embedding and truncation.

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
