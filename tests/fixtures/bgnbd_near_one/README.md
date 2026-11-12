# BG/NBD near-unit dropout shape

The 22 fixed synthetic histories use explicit constants and no random generator. Each history has four prediction horizons. The inputs exercise dropout shape one and both signs of offsets from 1e-4 through 1e-12. JSON numeric inputs denote their exact binary64 values, pinned by the shape hex field.

The reference integrates the active posterior: gamma purchase rate with shape r+x and rate alpha+T, and beta dropout probability with parameters a and b+x. Conditional on those latent values, expected future purchases are (1-exp(-lambda*p*h))/p. Integrating the purchase rate gives the one-dimensional beta-posterior integral used by generate.py. This integral is finite and continuous at a=1. The validator uses the independent Euler-transformed hypergeometric identity, differentiating its analytic numerator at a=1.

Primary source: Peter S. Fader, Bruce G. S. Hardie and Ka Lok Lee, 2005, "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model, Marketing Science 24(2), 275-284, DOI 10.1287/mksc.1040.0098. Only mathematical formulas are used, with original prose and made-up inputs.

Two separate CPython processes with mpmath 1.3.0 evaluated the posterior integrals at 60 and 90 digits. Their rounded JSON bytes match exactly. No native library was called to obtain these answers. The validator checks a different special-function expression at 80 digits. Tools require Python and mpmath only.

Run python3 -B generate.py 90 and redirect stdout to a file outside this directory to replay the key. Run python3 -B validate.py to check the existing bytes without writing. The allowance is absolute 1e-12 plus relative 1e-10. Zero horizons have exact zero means. The allowance concerns ordinary binary64 evaluation, not model uncertainty.

No fitted parameters, covariance, interval endpoints, arbitrary extreme-range accuracy or global domain certification is supplied.
