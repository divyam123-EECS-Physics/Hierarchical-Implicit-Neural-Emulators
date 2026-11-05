# Cursor prompt: HINE scalar roots and Hypothesis H

Implement evaluation tools for my existing HINE project, with a local Py2D repository available. Complete the work below in small stages, beginning with L=2. Inspect the repository before choosing file paths or APIs. Do not retrain models or change their mathematical update.

The intended HINE code is on `jacobian_eigenanalysis_stability_1`, not `main`. Check the current branch and working tree first. Use a separate checkout if necessary to inspect that branch without disturbing uncommitted work. Inspect applicable AGENTS.md instructions. Find the existing checkpoint loader, data loader, normalizers, downsampling operators, finite-difference/Jacobian-action code, and Arnoldi implementation. Reuse them where correct. Do not assume a particular Py2D fork, integrator API, HINE head name, or tensor layout.

The attached note “Is HINE Implicit?” provides the mathematical target. Its order-of-work table calls scalar roots Task 3, implicit closure/DB Task 4, and synchronization/Hypothesis H Task 5. Equation (14.1) is the fine-block identity; Hypothesis H is actually in Section 14.3. The relevant equations are reproduced below so this prompt is self-contained.

## Scope and conventions

- Analyze existing direct-step HINE checkpoints. Sections 8–10 of the note introduce an integration-constrained comparison variant; do not assume my checkpoint implements that variant. No tendency conversion is needed for these experiments.
- Do not construct dense fine-grid Jacobians. Do not call `torch.jvp`, `torch.func.jvp`, or silently substitute a new autodiff backend. Prefer the repository's existing matrix-free derivative implementation. If unavailable, implement central directional finite differences as described below.
- Keep injection and retention decompositions outside this task.
- Work with one reference example at a time during differentiation. Set the model to evaluation mode and disable stochastic behavior.
- Use physical vorticity coordinates for composing model and solver operators. Wrap the trained model's actual input normalization and output decoding so derivatives map physical input perturbations to physical output perturbations. Respect any residual-output formulation already used in the repo. In particular, do not multiply a standardized latent-to-fine derivative by a physical Py2D tangent vector without the appropriate conversion.
- Distinguish `dt_dns`, the solver microstep, from `delta_t_model`, the interval represented by one HINE step. Integrate Py2D over the latter interval for physical flow derivatives. Verify the interval is represented exactly; never silently truncate an integer division.
- Infer physics from the checkpoint's dataset metadata. The supplied slides use 256x256, a 2pi-periodic domain, Re=1000, beta=20, alpha=0.1, dt_dns=1e-4, and snapshot interval 0.01. The note describes the published HINE setting as Re=1e4 with snapshot interval 0.05. These are different settings. Do not silently mix them or invent missing forcing parameters.
- Record checkpoint, repository revisions, reference sample/time, actual physics, preprocessing, projection, precision, finite-difference amplitudes, all time intervals, and random seeds in every run.

## 1. Shared adapters

For L=2, expose the actual complete physical-coordinate model step

    (u_next, z_next) = model_step_physical(u, z)

where input z represents D u_{n+1}, and the output latent represents D u_{n+2}. Verify these offsets against the dataset and training code. Here g is the fine output and h is the latent output.

Implement the actual linear restriction D and a documented right inverse Dplus with D(Dplus(z))=z. Define P=Dplus D and Q=I-P. For nonoverlapping average pooling, Dplus repeats the coarse value over its block. Check the actual architecture before using that choice. If D has no supported right inverse, report this explicitly rather than substituting an unrelated projector.

Expose two separate physical solver operations:

1. `advance_physical(u, interval)`: ordinary, unrestricted Py2D evolution.
2. `advance_conditional(u, reference_path, projection, interval)`: evolution with the prescribed reference coarse path, for Hypothesis H only.

These should operate in memory and avoid the file-saving production wrapper in inner loops. Reuse the local solver's verified numerical kernels. Validate a short unrestricted run against its existing implementation.

Inspect solver state carefully. Upstream Py2D's solver uses previous convection/state information in an Adams–Bashforth update. My local fork may differ. Preserve required state within intervals. A physical map Phi(u) from the current field alone needs an explicit, consistent startup policy and a time-step convergence check. If instead differentiating an augmented numerical state with stored history, define its embedding and perturb the history consistently; do not hold arbitrary old fields fixed and call that derivative dPhi/du. Recompute derived fields/caches after perturbing or clamping a state. Do not restart the whole solver at every DNS microstep.

For central directional finite differences, normalize a nonzero direction before choosing a relative perturbation amplitude and restore its norm afterward. For example, with a documented RMS norm, q=v/rms(v), h=eps_rel*max(rms(x),reference_scale),

    Jv ~= rms(v) * (F(x+h*q)-F(x-h*q))/(2*h).

Handle the zero direction explicitly. For a real operator acting on a complex Arnoldi vector, use J(Re(v)) + i J(Im(v)); do not feed complex-valued physical states into the model. Test approximate linearity and repeat representative actions across an epsilon sweep. Enable float64 for the physical solver when supported. Select model precision and epsilon from convergence behavior rather than choosing an arbitrarily tiny epsilon in float32.

## 2. Scalar roots: exact toy verification

The note's single-mode model is

    g(u,z)=alpha_s*u+beta_s*z
    h(u,z)=gamma_s*u+kappa_s*z
    a=exp(mu*delta_t_model)

with D=1 and exact on-graph fit:

    alpha_s=a*(1-beta_s)
    gamma_s=a*(a-kappa_s).

The augmented matrix has roots

    physical root: a
    parasitic root: s=kappa_s-a*beta_s.

Verify its characteristic polynomial and roots numerically against the explicit 2x2 matrix. Use scalar suffixes to avoid confusing these coefficients with the physical drag and beta-plane parameters.

Test beta_s=1,kappa_s=0 -> s=-a; beta_s=1,kappa_s=a -> s=0; beta_s=0 -> s=kappa_s. Include one exact-fit example with abs(s)>1. For abs(a)<1 the -a mode decays in absolute magnitude, but is not damped relative to the physical root. Do not report it as absolutely undamped.

This toy does not require a trained network or a Py2D run.

## 3. Scalar roots: trained L=2 model

At a true input pair (u_n,z_n), with z_n=D u_{n+1}, define

    B_n v = partial_z g(u_n,z_n)[v]
    K_n v = partial_z h(u_n,z_n)[v].

Hold u_n fixed and perturb only the latent. If using finite differences, obtain B_n v and K_n v from the SAME two full model calls. B_n v lives on the fine grid; K_n v lives on the latent grid.

The matrix counterpart of the scalar parasitic root is

    S_n v = K_n v - D [Phi'_{n+1}(B_n v)].

This is Equation (7.8). The physical tangent is evaluated at TRUE u_{n+1}, for an unrestricted evolution interval from t_{n+1} to t_{n+2}. It is not evaluated at u_n, and not at the model-predicted u_{n+1}.

Compute the tangent action by the validated solver derivative or central differences of `advance_physical`:

    Phi'_{n+1} w ~=
      [Phi(u_{n+1}+epsilon*w)-Phi(u_{n+1}-epsilon*w)]/(2*epsilon),

with direction/amplitude scaling handled by the shared derivative helper. Do not clamp coarse scales in this calculation.

Provide a square, matrix-free operator on the latent space. A 32x32 latent gives dimension 1024; the fine grid is an intermediate vector, never a dense matrix. Feed this callable into the existing Arnoldi implementation, or `scipy.sparse.linalg.LinearOperator` with nonsymmetric `eigs(..., which='LM')` if the existing implementation is unsuitable. Do not use `eigsh` without establishing symmetry.

Begin with a few held-out reference states and roughly 6–10 requested Ritz pairs, configurable through the CLI. Save real and imaginary parts of the Ritz values, vectors or a reproducible option to save them, residuals, convergence flags, and a clearly labeled estimated largest modulus. Check stability of the leading values against Krylov dimension, epsilon, and solver time step on a small representative subset. A partial Ritz spectrum does not certify that every eigenvalue is inside the unit disk.

Plot the complex values with the unit circle, and the estimated largest modulus across reference times/checkpoints. Label them as parasitic-block eigenvalue estimates. Fourier modes generally couple in nonlinear turbulence, so do not assign an independent scalar a_k to every Fourier mode or call a projected Rayleigh quotient an exact root. Optional modal visualizations should show the Fourier content of the actual computed eigenvectors.

Frozen spectra are local diagnostics. Add a product option along a chronological reference segment:

    w = S_n v
    log_gain_n = log(norm(w))  # v has unit norm
    v = w/norm(w).

After a direction-alignment burn-in, report sum(log_gain_n)/sum(delta_t_model) as a finite-time estimate of the leading growth rate for products of S_n. Handle zero/underflow explicitly and repeat with independent starts when needed. Do not equate an average of individual spectral radii to this product growth rate. Because derivative-fit coupling to the physical block can remain, these diagnostics alone do not prove stability of the full HINE rollout.

### Reuse from Task 4

The already-computed B_n v also gives

    DB_n v = D(B_n v).

Offer this diagnostic as an optional inexpensive companion. It requires no Py2D derivative. Keep DB_n, K_n, and S_n distinct. `K_n-DB_n` may be reported only as a separately labeled approximation using Phi' approximately I; it is not the exact S_n.

For DB, eigenvalues near 1 indicate potential difficulty with the implicit closure through I-DB. Eigenvalues near 0 alone do not establish that g ignores the latent: QB may still be large, and nonnormal operators can have large responses. Do not add an implicit solver or make stronger claims from that spectrum.

## 4. Hypothesis H, first experiment: solver synchronization

The physics question is whether fine differences decay when the two simulations receive the identical coarse trajectory at every time.

Implement a sharp spectral projector P_kc and Q_kc=I-P_kc for a cutoff sweep. State whether the retained set is radial or square, the inequality at the cutoff, and the wavenumber units. Use the local FFT layout and preserve real-valued fields, Hermitian symmetry, the mean-mode convention, and dealiasing. A square cutoff matches a coarse grid's componentwise Nyquist convention; a radial cutoff is a different experiment. Never silently mix them.

Use representative, spun-up reference states with the same physics as the checkpoint data. Evolve a reference run and a perturbed follower with identical forcing. Initialize

    follower(0) = reference(0) + epsilon * Q_kc(random_real_field),

normalizing the perturbation and verifying that its coarse projection vanishes. Evolve both at dt_dns. At every solver microstep replace the follower's coarse part with the reference coarse part at the MATCHING time, leaving its fine part unchanged:

    follower <- P_kc(reference) + Q_kc(follower).

For multistage methods, impose the corresponding reference coarse values at the stages; for multistep methods, keep the coarse history and dependent caches consistent at their respective times. Projecting only every HINE step is a different, sparsely observed experiment and must not be used as the continuous-clamping result. A microstep projection approximates continuous clamping; demonstrate convergence when dt_dns is reduced.

Run reference and follower together or regenerate short reference intervals. Do not require saving the entire fine-resolution trajectory at every DNS microstep to disk.

Record the fine-error amplitude

    E(t)=norm(Q_kc(follower(t)-reference(t))).

Provide semilog E(t)/E(0) plots. For a rough initial estimate, fit log E versus physical time only after transients and before saturation or roundoff. Record the fitting window. If logging squared error/energy, its log slope is twice the amplitude growth rate.

For the production conditional-growth estimate, periodically renormalize the fine perturbation around the current reference. If epsilon_j is the amplitude at the start of a block of duration tau_j and E_j the end amplitude, accumulate

    lambda_c ~= sum_j log(E_j/epsilon_j) / sum_j tau_j.

Use burn-in to align the direction, a small-perturbation convergence sweep, multiple initial directions/reference segments, and blockwise variability estimates. For a multistep implementation, renormalize the full perturbation history consistently or use the documented state-only interval adapter with a verified restart/convergence policy. Do not rescale only the displayed current field while leaving arbitrary old follower states in its history.

Suggested first cutoffs: 4, 8, 16, 32, 64. Include an unclamped control, then refine near a sign change, e.g. 12, 20, 24. Extend below 4 if all measurements are negative and above the tested range if necessary. The hypothesized threshold near 16 is a prediction to test, not an expected answer to enforce.

Report a bracket for the smallest tested cutoff with reliably negative growth, the uncertainty, and any insufficiently converged cases. Make a plot of lambda_c against cutoff with a horizontal zero line. Report exp(lambda_c*delta_t_model) as the equivalent average factor over one model step. This is an asymptotic/finite-time product-growth quantity, not a claim that every direction contracts at every step.

For a 256x256 field the note's nominal mapping is r=2,4,8,16 -> coarse side 128,64,32,16 -> kc=64,32,16,8. This mapping is approximate for average pooling. The exact solver cutoff geometry matters too.

## 5. Hypothesis H, second experiment: compare HINE with the conditional solver

The actual trained-model hypothesis is

    A_QQ,n approximately Psi_n,

where A_n=partial_u g(u_n,D u_{n+1}) with the latent input fixed, and Psi_n is the fine tangent map when the complete coarse reference path is prescribed throughout [t_n,t_{n+1}]. A solver-only threshold estimate does not establish this identification.

For a direct model comparison, use the SAME projection for HINE and Py2D. If HINE uses average pooling, run a separate conditional experiment with P=Dplus D and Q=I-P. Apply that pooling projector to the physical field inside the solver adapter. Do not equate a sharp Fourier experiment with an exact pooling experiment. Check projection/solver consistency and report it.

At a true input pair, choose v=Qv and compute

    a_v = Q partial_u g(u_n,D u_{n+1})[v],

holding the latent identical in both perturbed evaluations. This is A_QQ v. Do not regenerate the latent from the perturbed fine input, as that would measure a total derivative instead of the fixed-latent derivative.

Compute p_v=Psi_n v using two conditional solver runs initialized at u_n +/- epsilon*v. Supply the same reference coarse PATH to both, including all microsteps/stages, and finite-difference their fine outputs. Use exactly delta_t_model, the same projection, and the same perturbation direction/norm as for a_v.

For multiple true times and projected directions, save:

- norm(a_v)/norm(v) and norm(p_v)/norm(v);
- norm(a_v-p_v)/(norm(a_v)+norm(p_v)+small_floor), with the floor recorded;
- directional cosine when both outputs are sufficiently nonzero;
- variation under epsilon and dt_dns changes.

Plot paired gains and action mismatch across reference states. Matching gains alone is weaker evidence than matching vector actions. If long-time factors are compared, compute separately renormalized products of A_QQ,n and Psi_n along the same reference sequence and use the same physical time/norm convention. Do not compare an instantaneous eigenvalue or an arithmetic mean of one-step norms directly with exp(lambda_c*delta_t_model).

The primary common norm should be physical vorticity RMS/L2. The note's energy inequality uses velocity L2; optionally log velocity differences through the solver's vorticity-to-velocity conversion, but do not equate their finite-time norms. Keep the model/solver comparison in identical coordinates and norms.

This experiment tests a hypothesis. HINE receives coarse endpoints, whereas the conditional solver receives the entire coarse path. The discrepancy can matter, particularly at larger model time steps. Do not force agreement or describe synchronization alone as proof of HINE stability.

### Optional Equation (14.1) diagnostic

Only after the direct H comparison works, optionally expose the actions in

    A_QQ = Q Phi'_n Q - Q B_n D Phi'_n Q + Delta_g,QQ,
    Delta_g = A_n + B_n D Phi'_n - Phi'_n.

Here the physical tangent is evaluated at u_n, unlike the u_{n+1} tangent in S_n. For v=Qv, compute f=Phi'_n v with the unrestricted solver, qf=Qf, qb=Q B_n(Df), and defect=a_v+qb-qf. Save their norms and alignments. The identity is exact by definition of Delta_g, so a small algebraic residual checks the implementation; it does not validate Hypothesis H. The size of the derivative-fit defect and the comparison A_QQ versus Psi are scientifically informative.

## 6. L=3 handling

Deliver a working L=2 path first. The note explicitly derives L=2 only. Do not silently analyze an L=3 checkpoint as if it had a single latent.

If the local code supports L=3 and verifies inputs (u_n,D1 u_{n+1},D2 u_{n+2}), define z as the stacked latent, B=partial_z g, and K=partial_z h, where h stacks both future latent outputs. Define the physical consistency graph R(u)=(D1 Phi(u),D2 Phi(Phi(u))). The derived extension is

    S_n v = K_n v - R'(u_{n+1}) B_n v.

For w=B_n v:

    w1=Phi'_{n+1} w
    w2=Phi'_{n+2} w1
    R'(u_{n+1})w = stack(D1 w1, D2 w2).

Both physical tangent actions use ordinary evolution and true trajectory states. Validate each actual latent time offset and restriction in the code, since D2 need not be the square of D1. If offsets differ, derive the matching graph rather than using these formulas blindly. Label this as an extension of the note.

For fine-block diagnostics in L=3, hold both latent inputs fixed and define the Q split explicitly. Treat comparison to a conditional solver as an additional hypothesis for that architecture, not a result already derived in the note.

## 7. Deliverables and verification

Fit the following capabilities into the repo's existing conventions, with documented CLI arguments:

1. Scalar toy check.
2. L=2 parasitic-block Arnoldi analysis and optional DB companion.
3. Py2D cutoff/synchronization screening, usable without any checkpoint.
4. Matched-projection A_QQ versus Psi comparison with a checkpoint.

Suggested knobs include checkpoint, data/reference selection, Py2D path, physics config, model interval, DNS step, projection type, cutoff/r sweep, perturbation amplitude, random seed, sample count, horizon, renormalization interval, Arnoldi size, tolerance, and output path. Do not silently choose absent physical metadata.

Save structured results (NPZ and JSON/CSV as appropriate), plots, and sufficient provenance to reproduce each result. Report failed convergence as failed/inconclusive, not as a stable eigenvalue estimate. Keep filenames scoped by experiment and checkpoint so existing results are preserved. Provide smoke commands and longer evaluation commands using the actual implemented filenames. Run the smoke work supported by available data; if checkpoints/data/hardware are missing, finish the implementation and state exactly which numerical validations remain unrun.

Use meaningful checks:

- Exact 2x2 toy roots, including a growing parasitic mode despite exact on-graph fit.
- A small coupled linear fixture verifying the matrix-free S action and the n+1 time index against explicit small matrices.
- Projection identities and preservation of the prescribed coarse component through the conditional run.
- A coupled linear example where unrestricted Q Phi' Q differs from the continuously conditioned fine map, to catch endpoint-only clamping.
- Directional finite-difference consistency, epsilon convergence, and DNS time-step convergence on representative actions.
- Replay of the unrestricted solver adapter, and renormalization/history correctness for the actual integrator.

Explain the discovered model indexing, how numerical solver state is handled, commands to run, and the meaning of each plot. Never present a predicted threshold, successful HINE/solver match, full stable spectrum, or expensive numerical result unless it has actually been measured.

## Source references

- User-supplied HTML: Sections 2, 5–7, 12, 13, 14, 15 and assumptions A4–A7 in Section 16.
- User-supplied step-size presentation: experimental testbed on slide 13 (printed slide number 14).
- Upstream Py2D implementation, to compare with the local fork: https://github.com/envfluids/py2d/blob/main/py2d/Py2D_solver.py
- SciPy nonsymmetric matrix-free eigensolver: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
- Conditional synchronization methodology: Inubushi and Caulfield, “Synchronisation in two-dimensional damped-driven Navier–Stokes turbulence: insights from data assimilation and Lyapunov analysis,” https://doi.org/10.1017/jfm.2025.11057 . Its flow settings do not establish the threshold for my beta-plane dataset.
