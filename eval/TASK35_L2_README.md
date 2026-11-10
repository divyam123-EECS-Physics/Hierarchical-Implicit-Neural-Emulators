# Tasks 3 and 5 — L=2 parasitic roots and Hypothesis H

The network is the trained direct-step map `ema_vit(u, [z], t=1) -> (g, h)`.
Nothing is rewritten as a tendency model. These scripts do not touch Task 1.

## Indexing

An L=2 step takes `(u_n, D u_{n+1})` and returns `(u_{n+1}, D u_{n+2})`.
`D` is average pooling with the checkpoint's `ds_factors[0]` (8 for the
256 to 32 run). Model tensors are `Omega / --norm`. Jacobian actions are
converted back to physical vorticity, so the factor `norm` cancels.

Eval `.mat` files are 0.01 apart. `--every-nth-field 5` makes one model
step `delta_t_model = 0.05`, which is 500 solver steps of `dt_dns = 1e-4`.
The runners refuse to continue if the time stamps disagree with `--delta-t`.
Task 1's label `dt = 0.25` is five of these model steps, not one.

The 256^2 eval frames are spectral truncations of the 512^2 DNS in

`/pscratch/sd/d/divyam/py2d_runs/paper/results/Re10000_fkx4fky4_r0.1_b20/NoSGS/NX512/dt0.0001_IC1`

with the same integer stems. Physics, from that run's `parameters.txt`:
Re=10000, fkx=fky=4, alpha=0.1, beta=20, no forcing filter, dealiasing on.
`Phi'` is the unrestricted 512^2 map. `S_n` evaluates it at the true
`u_{n+1}`, over `[t_{n+1}, t_{n+2}]`.

## Solver state

`eval/py2d_inmem.py` steps in memory with the local py2d kernels.
Convection is Adams-Bashforth; diffusion and drag are Crank-Nicolson.
A single-field start sets the previous and current vorticity equal and
builds the convection history from that field. That is the solver's own
first iteration, and it is not an exact restart. An exact restart uses
`from_hats` on a saved `IC/*.mat` pair and recomputes `convec0` from
`Omega0_hat`.

`--replay-check` reports three numbers:

- `exact_rel_rms`: IC/104999 advanced 0.01 against data/105000. This is the
  match to the stored integrator.
- `startup_rel_rms`: data/91836 advanced from the physical field alone
  against data/91837. History was dropped on purpose.
- `half_dt_rel_rms`: the same startup interval at `dt_dns = 5e-5`.

Conditional runs replace the coarse part after every microstep and
recompute streamfunction and velocity from the projected vorticity on the
next step. The Adams-Bashforth history is therefore the history of
projected states. Renormalization scales the fine difference at both
stored time levels and rebuilds `convec0`.

## Projectors

Sharp cutoffs use integer wavenumbers on the 2-pi box and a strict
inequality. Square means `|kx| < kc` and `|ky| < kc`. Radial means
`|k| < kc`. `kc = 0` is the unclamped control, not a cutoff.
The pooling projector is `pool_8(truncate_512_to_256)` with the blockwise
repeat as the right inverse. It is the projector used for `A_QQ` versus
`Psi`. It is not the sharp cutoff.

## Commands

Smoke (what `eval_task35_l2_debug.sbatch` runs):

```bash
sbatch nersc_job_launchers/eval_task35_l2_debug.sbatch
```

Toy only:

```bash
python eval/task3_toy_roots.py --out /tmp/toy_roots.json
```

Parasitic block, 20 states, 8 Ritz values:

```bash
python eval/task3_parasitic_arnoldi.py \
  --n-states 20 --k 8 --ncv 24 --maxiter 120 --db \
  --save-dir /pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task3_parasitic/n20
```

Add `--product-steps 8 --burn-in 2 --n-starts 2` for the renormalized
product of `S_n`. Add `--stability-subset 2` to repeat the first two
states over `ncv`, epsilon, and `dt_dns`. Add `--n-gpus 4` to split states.

Synchronization screen:

```bash
python eval/task5_sync_screen.py \
  --kc 0 4 8 12 16 20 24 32 64 --T 2.0 --n-segments 3 --n-directions 3 \
  --also-dt 5e-5 --n-gpus 4 --skip-existing \
  --save-dir /pscratch/sd/d/divyam/hine_runs/py2d_sync/square_kc
```

`--n-gpus 4` gives each GPU the tasks whose index modulo 4 equals its rank.
`--skip-existing` leaves a finished `kc*_seg*_dir*.npz` in place. The summary
and both plots are written after every rank has finished.

`A_QQ` versus `Psi`:

```bash
python eval/task5_hine_vs_conditional.py \
  --n-states 8 --n-directions 4 --stability-subset 2 \
  --n-gpus 4 --skip-existing \
  --save-dir /pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task5_hine_vs_cond/n8
```

Cases are split the same way, one `(state, direction, epsilon)` per file,
and `comparisons.npz` is written after the ranks join.

## Plots

- `ritz_argand.png`: Ritz values of `S_n` and the unit circle. A partial
  spectrum. Failed ARPACK runs stay flagged `converged=False` in the npz.
- `largest_modulus.png`: estimated leading modulus at each reference time.
  It is not a proof that the whole spectrum is inside the unit disk.
- `leading_vector_fourier.png`: log power of the leading computed vector.
  Not an independent root of each Fourier mode.
- `E_over_E0_semilog.png`: fine-error amplitude in the first
  renormalization block.
- `lambda_c_vs_kc.png`: finite-time conditional growth rate. The zero line
  is the sign change being measured. `step_factor` in `summary.json` is
  `exp(lambda_c * delta_t_model)`.
- `paired_gains.png` and `mismatch_cosine.png`: `||A_QQ v||` against
  `||Psi v||`, and the vector mismatch. Matching gains are weaker than
  matching actions.

## What the 30-minute smoke does not cover

It does not run the 20-state Arnoldi, the product of `S_n`, the ncv/epsilon
grid, the cutoff refinement around a sign change, or a long `dt_dns`
convergence study. Those are the commands above. `S_approx_phi_identity`
in the npz is `K - DB`, saved only as that approximation.
