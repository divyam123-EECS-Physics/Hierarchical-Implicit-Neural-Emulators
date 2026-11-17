# L=2 energy diagnostics

The definitions follow the workspace note `README_HINE_Energy_Stability.md`. That note was not in this repository when the diagnostics were added. Two names in the note are changed here so they do not collide with existing HINE notation:

- The note's latent self-Jacobian `D_n` is `K`. `D` is only average pooling.
- The note's one-step residual `r_H` is `epsilon_H`. Task 1 retention channels `r_H` and `r_Z` are not used.
- The note's input Gram operator `Q_n` is `Q_E`. `Q` is the fine-error energy `0.5 <e, G e>`.

Blocks are `A = dF_H/du`, `B = dF_H/dz`, `C = dF_Z/du`, and `K = dF_Z/dz`. The budget applies `A` and `B` only.

## Staggered L=2 step

The direct step takes `(u_n, z_{n+1})` and returns `(u_{n+1}, z_{n+2})`, with `z_{n+1} = D u_{n+1}`. A free rollout starts at `(u_0, D u_1)`. `A` and `B` are evaluated at the teacher-forced pair `(u_n, D u_{n+1})`. The directions are the free-rollout errors at those inputs.

## Energy

`E(u) = 0.5 <u, G u>` matches `kinetic_energy_torch`. `G` is the inverse Laplacian on nonzero modes, followed by the same inverse real FFT. That transform drops the imaginary part of the Nyquist velocity, so the quadratic form is the physical velocity inner product `mean(u_f u_g + v_f v_g)`, not the raw sum of `invKsq |omega_hat|^2`. The mean mode stays at `invKsq[0, 0] = 0`. The Nyquist column of `invKsq` is kept.

`mu_E` and `sigma_E` are read from the existing reference cache over stems `10000..105000`. They are not recomputed from predictions.

Shells use `k_bin = rint(|k|)` and rFFT multiplicity 2 on interior `kx`. Bands are `0 < |k| <= 16`, `16 < |k| <= 64`, and `|k| > 64`.

## Weighted directions

Input coordinates use physical RMS: `v_phys = sqrt(n_block) v_coord`. Output coordinates use `pack`, with `||pack(y)||^2 = <y, G y>`. `M_HZ` is the composition of those weighted maps, `Ã* B̃`. `Q_E` is the Gram operator of `[Ã B̃]`. Negative `C_HZ` is quadratic cancellation in the realized errors. It is not, by itself, evidence that the latent pathway causes `d_E <= 5`.

## Shared helper left unchanged

`ModelStepPhysical._out` in `eval/l2_model_jvp.py` casts the network output to float32 before scaling by `norm`. These diagnostics do not call it. The float64 adapter is `eval/energy_l2_model.py`.
