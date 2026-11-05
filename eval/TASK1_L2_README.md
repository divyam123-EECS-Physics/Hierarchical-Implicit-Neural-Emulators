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

## Metric → equation

| output | meaning | equation | status |
|---|---|---|---|
| `tr_A_*`, `tr_A_over_N` | Hutchinson trace of `A = dg/du` | (5.3), (9.2) | exact up to FD and probe noise |
| `tr_K_*` | trace of `K = dh/dz` | (5.3) | exact up to FD |
| `tr_DB_*`, `tr_DB_over_N` | trace of `D B` from coarse probes | (9.2) second term without the `1/dt` | exact up to FD |
| `tr_BD_paired_*` | `v·B D v` on the same fine probes as `A` | `Tr(BD)=Tr(DB)` | exact up to FD |
| `tr_J_*`, `tr_J_over_N` | trace of `J_phys` or `J_phys_proxy` | (7.2) | exact only if `--physical-jvp py2d` |
| `tr_Fu_over_N` | `(Tr(A) - N) / (N dt)` | (8.2), (9.2) | difference-quotient label |
| `tr_DFz_over_N` | `Tr(DB) / (N dt)` | (9.2) | difference-quotient label |
| `tr_Heff_over_N` | `(Tr(J) - N) / (N dt)` | (8.3), (9.2) | difference-quotient of `J`, proxy unless py2d |
| `tr_S_*`, `tr_S_over_M` | trace of `S` or `S_proxy` | (7.8) | exact only with py2d |
| `tr_Sigma_over_M` | `(Tr(S) - M) / (M dt)` | (8.4), (9.3) | difference-quotient label |
| `amp_J_*` | `‖J v‖² / N` | Section 9, direct-step contraction | not a trace |
| `offgraph_probe_*` | `‖L^off v‖² / N` | (7.7) lower-left, (6.3) | exact only with py2d |
| `identity_gap` | `Tr(J) - Tr(A) - Tr(BD)` on paired probes | proxy identity | should be ~0 for a linear map |
| `rho_S` | leading `‖λ‖` of `S` | Section 7.6 | optional `--spectral-radius` |
| `roll_eps_u_norm`, `roll_eps_z_norm` | teacher-forced one-step residuals | (5.2) | exact |
| `roll_e_norm` | `‖û_n - u_n‖` on a free rollout from `(u_0, D u_1)` | (5.1) | exact |
| `roll_I`, `roll_R` | `I_n=‖∑ ε^u‖`, `R_n=‖e_n‖-I_n` | (10.2) and the deck | exact bookkeeping |
| `roll_rho_norm` | `‖e_n - ∑ ε^u‖` | (10.4) | exact; not the same as `R_n` |
| `roll_rH_norm`, `roll_rZ_norm` | first-factor channels | (10.4) with `dt H → J-I`, `dt F_z → B` | approximation |
| `roll_recon_norm` | `‖ρ - r_H - r_z‖` | leftover past one factor | approximation |
| `roll_cos_rH_rZ` | alignment of the two channels | Section 10 | so the norms are not added |

`approximation` in `results.npz` is `proxy` or `exact`. Proxy means `D Φ' ≈ D`.
Do not read proxy traces as the Section 7 operators.

## What still needs the solver

`J_phys`, `S`, `L^off`, and `ξ = ε^z - D Φ' ε^u` are the Section 7 quantities
only with `--physical-jvp py2d`. That path is `Py2DFDTangent` in
`eval/l2_physical_jvp.py` (AB2/CN, NoSGS, Re=1e4, fkx=fky=4, α=0.1, β=20,
solver dt=1e-4, emulator dt=0.25). It imports jax lazily and raises if jax
is missing. The `hine` env is not modified. Use
`nersc_job_launchers/make_hine_jax_env.sh`.

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
