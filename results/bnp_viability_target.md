# BNP viability target: analytical screening criterion

## Existing configuration and evidence

- In loads.py, each class archive has shape (members, days, T).
  fit_load_model flattens members and days into observations:
  m_class = len(classes[class]) * n_days.
- The BNP feeder runners enumerate OpenDSS Load objects for their ratings.
  Records therefore represent load-object-days; physical customer-days
  require a verified mapping from customers to those objects.
- assign_classes sorts ratings and splits them into three nearly equal groups.
  The local IEEE 123 model defines 91 Load objects: classes of 31, 30, 30.
  With the runners' 90 historical days, m is 2790, 2700, 2700.
- T defaults to 96 (15-minute samples). The feeder BNP runners use C=6.
  The privacy functions default to clip_norm=None, resolved as sqrt(T)*R/2.
- R is the class-specific LOG-space truncation width:
  R = log(p_max) - log(p_min).
  fit_load_model uses p_min=min(archive_class)/1.5 and
  p_max=1.5*max(archive_class), so R=log(2.25*max/min).
  Exact per-class R values are not recorded in the existing result files.
  They have not been regenerated or estimated here.
- Reference signal s=0.0154 comes from results/bnp_peer_review.md.
  Its generating statistic is median(abs(np.cov(log_records, rowvar=False)))
  for class 0, across all matrix entries. Other classes' values are unreported.

## Derivation

Let delta denote the argument passed to bnp_fit_class.
The implementation allocates delta/2 each to mean and covariance.

    S_cov = 2*C^2/m
    B_cov = S_cov/(2*(delta/2)) = 2*C^2/(m*delta)
    uniform_sd = B_cov/sqrt(3)
    m_required = ceil(2*C^2/(sqrt(3)*delta*s))

At m=2790, C=6, delta=0.02, this gives B_cov=1.2903 and
uniform_sd=0.7450, consistent with the saved report.
Symmetrization leaves diagonal sd=B/sqrt(3); off-diagonal sd=B/sqrt(6).
The table uses the requested uniform sd, before covariance eigenvalue repair.

## Acceptance table

All counts are PER CLASS. The customer column assumes one independently
represented customer per load object; otherwise interpret it as load objects.
Integer thresholds use the existing rounded s=0.0154, not a new measurement.

| delta | required m | customers x days sufficient | T | C |
|---|---:|---:|---|---:|
| 0.02 | 134,965 | 1,500 x 90 | 96 or 24 | 6 |
| 0.05 | 53,986 | 600 x 90 | 96 or 24 | 6 |
| 0.10 | 26,993 | 300 x 90 | 96 or 24 | 6 |
| 0.02 | 33,742 | 375 x 90 | 96 or 24 | 3 |
| 0.05 | 13,497 | 150 x 90 | 96 or 24 | 3 |
| 0.10 | 6,749 | 75 x 90 | 96 or 24 | 3 |

## Interpretation and limits

- At fixed C and s, reducing T alone does not change covariance-noise sd.
  Mean-noise sd is sqrt(T)*R/(m*delta*sqrt(3)); hourly T halves it at fixed R.
- Choosing C proportional to sqrt(T) takes C=6 at T=96 to C=3 at T=24,
  reducing required m fourfold if s is unchanged. This is a proposed scenario.
- With clip_norm=None, m_required=ceil(T*R^2/(2*sqrt(3)*delta*s)).
- Smaller C can introduce clipping bias. Changed resolution can change s.
  Neither effect has been measured. Simply setting T=24 also retains the
  generator's per-step AR coefficient; it is not hourly aggregation.
- More days increase record count under record-level adjacency; they do not
  automatically give the same privacy guarantee for a customer's whole history.
- Passing this table establishes noise-scale parity only. New-regime covariance,
  clipping bias, temporal fidelity and voltage utility remain unmeasured.
