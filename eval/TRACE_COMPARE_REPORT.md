# HINE trace comparison

Dots are probe-averaged Jacobian diagnostics at true inputs. They are not eigenvalue samples and not individual Hutchinson probes. Each category is one loader trajectory, so the median and interquartile interval are descriptive. Probe standard errors are in `metrics.npz` and are not the spread drawn on the plots.

Negative trace values are negative isotropic linear energy contributions. They do not prove contraction or long-horizon stability.

Wavenumber-4 L=1 versus L=2 is the controlled hierarchy comparison. Wavenumber 5 repeats that comparison on a different forcing. Cross-wavenumber differences are not attributed to hierarchy alone.

## L=1 · k_f=4

Direct trace median -0.9889, interquartile -0.9903 to -0.9881, fraction below zero 1.
Isolated energy change median -0.743, fraction below zero 1.
Latent diagnostics are N/A for this fine-only model.
Error-direction direct trace median -0.9531 (fraction below zero 1). Final free-rollout error 462.3. Retention R median 1.46.

## L=2 · k_f=4

Direct trace median -0.991, interquartile -0.9924 to -0.9899, fraction below zero 1.
Isolated energy change median -0.7574, fraction below zero 1.
Latent trace median -0.779. Frozen loop median 0.0002183. Consecutive loop median 0.000225. Loop net effect median 0.00338, fraction below zero 0.
A negative net effect means the latent loop reduces average two-step energy relative to the direct path. Negative interference alone only says the paths oppose.
Error-direction direct trace median -0.9563 (fraction below zero 1). Final free-rollout error 502.8. Retention R median 1.348.
Frozen latent gain medians by horizon: h=1 0.234, h=2 0.123, h=4 0.06369, h=8 0.02479. Time-varying gain medians: h=1 0.234, h=2 0.1271, h=4 0.07391, h=8 0.03072. A gain above 1 amplifies an isotropic latent perturbation. A negative latent trace does not forbid that.

## L=1 · k_f=5

Direct trace median -0.9802, interquartile -0.9843 to -0.9786, fraction below zero 1.
Isolated energy change median -0.6879, fraction below zero 1.
Latent diagnostics are N/A for this fine-only model.
Error-direction direct trace median -0.8686 (fraction below zero 1). Final free-rollout error 912.3. Retention R median 174.2.

## L=2 · k_f=5

Direct trace median -0.9781, interquartile -0.9823 to -0.9765, fraction below zero 1.
Isolated energy change median -0.6986, fraction below zero 1.
Latent trace median -0.7762. Frozen loop median 0.0003589. Consecutive loop median 0.0003629. Loop net effect median 0.004075, fraction below zero 0.
A negative net effect means the latent loop reduces average two-step energy relative to the direct path. Negative interference alone only says the paths oppose.
Error-direction direct trace median -0.9279 (fraction below zero 1). Final free-rollout error 873.8. Retention R median 126.9.
Frozen latent gain medians by horizon: h=1 0.1689, h=2 0.06145, h=4 0.01437, h=8 0.001067. Time-varying gain medians: h=1 0.1689, h=2 0.06257, h=4 0.01539, h=8 0.001134. A gain above 1 amplifies an isotropic latent perturbation. A negative latent trace does not forbid that.

## Answers from these states

The most negative direct-trace median is L=2 · k_f=4 at -0.991.
Its isolated energy-change median is -0.7574. Energy change equals twice the normalized trace plus the normalized squared shift, so a negative trace can still leave the isolated map energy-amplifying.
On these states, the sign of the direct trace and the sign of the isolated energy change agree for every setting.
L=2 · k_f=4 does not show a negative loop-net median. L=2 · k_f=5 does not show a negative loop-net median.
Realized fine errors occupy a negative-energy direction where the error-direction trace or energy change is negative. That local direction is compared with the free-rollout error above; growth of the rollout error means the one-step linear contribution does not determine the trajectory.
These local diagnostics do not establish long-horizon nonlinear stability.

Git commit recorded with the run: f70fc4d71cfd62931372be148f108bbafc3f4a8b. Analysis code may be uncommitted.

