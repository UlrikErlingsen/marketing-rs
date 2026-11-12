# BG/NBD forecasts at large r

The 87 fixed synthetic histories use explicit constants and no random generator. Together they have 419 prediction horizons. They cover the two reported wrong forecasts, a grid of r from 20 to 2000 with four dropout-shape pairs and three histories, a purchase rate of ten per week, nearly homogeneous purchase rates with r = 50000, large r with large x, ordinary fitted magnitudes, both sides of the 100000-term limit of the transformed series, both sides of the rounding guard, and two cases inside the near-one band. JSON numeric inputs denote their exact binary64 values.

The reference integrates the active posterior: gamma purchase rate with shape r+x and rate alpha+T, and beta dropout probability with parameters a and b+x. Conditional on those latent values, expected future purchases are (1-exp(-lambda*p*h))/p. Integrating the purchase rate gives the one-dimensional beta-posterior integral used by generate.py. The validator evaluates the published closed form, equation 10, with a high-precision hypergeometric function. The two routes share no series and no quadrature.

Primary source: Peter S. Fader, Bruce G. S. Hardie and Ka Lok Lee, 2005, "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model, Marketing Science 24(2), 275-284, DOI 10.1287/mksc.1040.0098. Only mathematical formulas are used, with original prose and made-up inputs.

Two separate CPython processes with mpmath 1.3.0 evaluated the posterior integrals at 60 and 90 digits. Their rounded JSON bytes match exactly. The validator checks equation 10 at 80 digits and requires the same binary64 value for every probability and mean, 1257 comparisons. No native library was called to obtain these answers. Tools require Python and mpmath only.

Run python3 -B generate.py 90 and redirect stdout to a replay file outside this directory. Run python3 -B validate.py to check the existing bytes without writing. The scalar allowance is absolute 1e-12 plus relative 1e-9. The relative part is the rounding tolerance that the forecast series must certify. The allowances concern ordinary binary64 evaluation, not model uncertainty.

The key states what the model's formula gives. It does not say which inputs an implementation must answer. An implementation may refuse an input whose rounding it cannot bound. The test that reads this key names every refused input. No fitted parameters, intervals or global domain certification is supplied.
