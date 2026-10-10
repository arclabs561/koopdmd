# koopdmd

Local draft spot. Not a crate yet.

Purpose: Koopman and dynamic mode decomposition operators: DMD, EDMD, kernel
DMD, and Koopman eigenfunction approximations.

Current verdict: keep this README-only. `rkhs/examples/kernel_dmd.rs` is the
first proof: it lifts scalar nonlinear dynamics into RBF features, fits a
one-step linear operator with ridge regression, and beats an identity feature
baseline. That belongs in `rkhs` until a second operator proof needs a shared
DMD/EDMD type and error surface.

First proof: fit a kernel DMD operator on a small nonlinear trajectory using
`rkhs`, then predict one rollout and report error.

Extraction rule: keep the first slice operator-focused. Do not turn this into a
broad control, neural Koopman, or forecasting framework.
