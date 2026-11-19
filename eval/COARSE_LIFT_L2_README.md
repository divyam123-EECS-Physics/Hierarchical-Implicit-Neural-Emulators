# L=2 coarse-lift diagnostics

This run measures how the latent head responds to fine vorticity, split into a block-constant lift and the unresolved remainder. It is not a retraining run, and it does not replace the Task 1 diagnostics.

The map is the trained direct step

`(u_n, z_{n+1}) -> (u_{n+1}, z_{n+2})`

with `z_{n+1} = D u_{n+1}` at teacher-forced states. The blocks are

- `A = dg/du`, `B = dg/dz`, `C = dh/du`, `K = dh/dz`.

`C` holds the latent input fixed. It does not differentiate a latent initialization through the fine field.

## Operators

For the `plus_L2_m32` checkpoint, `D` is nonoverlapping uniform average pooling by `r = 8` (`256 -> 32`). That is checked against `FutureINNPlus._pool_seq` before any spectrum is computed.

- `D†` repeats each coarse value over its `r×r` block. This is the Moore–Penrose pseudoinverse.
- `D*` writes that value divided by `r²`. It is the Euclidean adjoint, and it is not used as the lift.
- `D† = r² D*`.
- `P = D† D` and `Q = I - P`.

The lift-and-response operators are

- `T = C D†` (coarse to coarse),
- `U = D† C` (fine to fine),
- `C` (fine to coarse),
- `CQ` (unresolved fine to coarse).

`T` and `U` have the same nonzero eigenvalues. `U` has `N - M` extra zero modes. Right eigenvectors of `U` are `D†` applied to those of `T`. Left eigenvectors of `U` are `C*` applied to left eigenvectors of `T`. Singular values do not follow that correspondence, so `T`, `U`, `C`, and `CQ` are factored separately.

Left eigenvectors are the Arnoldi vectors of `T*`. For this real operator, `T* y = μ y` means `y` is a left eigenvector of `conj(μ)`. If the five-mode subspace splits a conjugate pair and the adjoint returns `μ = λ` rather than `conj(λ)`, the stored left vector is `conj(y)`. `left_values` are those matched adjoint eigenvalues, so a converged left pair has `left_values ≈ conj(λ)`.

If Arnoldi stops before five modes converge, the partial vectors are kept and the solve is repeated once with twice the Lanczos subspace and twice the iteration cap. A state file is marked complete only when all five requested eigenvalues are present together with the four singular-value sets.

## Units and gains

Computations are float64. The network sees vorticity divided by `norm` (10). The physical Jacobian equals that normalized Jacobian, and stored fields are multiplied back by `norm`. These blocks are not divided by `dt`. `dt = 0.05` is the spacing of the kept frames.

Euclidean gain is `||op v||_2 / ||v||_2`. The physical RMS operator is `C_RMS = sqrt(N/M) C`. `T` and `U` are square, so that normalization leaves their singular values unchanged. A gain on one rollout vector or one Rademacher probe is a sampled directional gain. It is not `σ_max`.

## What the numbers are not

`T` and `U` are lift-and-response probes. Neither is the autonomous HINE rollout. An eigenvalue inside the unit circle does not establish one-step contraction of the augmented map. Spectra at separate states do not establish stability of a time-varying product.

Coarse-lift sensitivity is `T`: the latent head's response to a fine perturbation that is constant on pooling blocks. Unresolved-fine sensitivity is `CQ`: the response to the part of a fine perturbation that averages to zero. Coupled feedback into the next fine state is the sum `A_{n+1} A_n e_H + B_{n+1} C_n e_H`, which is the fine component of the two-step augmented tangent when the initial latent perturbation is zero. `B_{n+1} C_n e_H` alone is only the path that re-enters through the latent.

## State selection

The sample manifest is the corrected Task 1 index

`/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/task1_l2_n20_float64_py2d_dt05/sample_index.json`

20 windows, 42 frames, rollout length 40. Each output directory copies that file and writes `selection.json`.

- Budget: all 20 windows, steps 0 through 39.
- Eigenvalues of `T` and singular triplets: manifest samples 0, 10, and 19 at steps 0, 20, and 39. Leading 5 modes.
- Two-step check: those windows at steps 0, 20, and 38. Step 0 has zero initial fine error, so that row is the zero identity.

## Commands

From the `coarse-lift-l2` worktree, with the `hine` environment:

```bash
python tests/test_l2_coarse_lift.py

sbatch nersc_job_launchers/eval_l2_coarse_lift_debug.sbatch

srun -A m5326 -C gpu -q interactive -t 02:00:00 -N 1 -n 1 -c 128 \
  --gpus-per-task=4 --job-name=hine_l2_clift_budget \
  bash nersc_job_launchers/eval_l2_coarse_lift_budget.sh

srun -A m5326 -C gpu -q interactive -t 04:00:00 -N 1 -n 1 -c 96 \
  --gpus-per-task=3 --job-name=hine_l2_clift_spectral \
  bash nersc_job_launchers/eval_l2_coarse_lift_spectral.sh
```

Outputs:

- `/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/coarse_lift_l2_smoke/`
- `/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/coarse_lift_l2_budget_n20/`
- `/pscratch/sd/d/divyam/hine_runs/plus_L2_m32/coarse_lift_l2_spectral_n9/`

Finished windows are not recomputed. Task 1 files and previous output directories are not modified.

## Shared helpers

`l2_fd_jvp.py`, `l2_operators.py`, `l2_retention.py`, and `coarse_projectors.py` are imported or left unread. They are not patched by this branch. `coarse_projectors.PoolingProjector` is a solver-grid spectral projector and is not `D†`. No shared-helper bug was found while adding this diagnostic.
