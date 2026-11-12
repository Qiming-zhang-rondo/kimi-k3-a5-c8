# VA0.30 Kimi K3 QFM-only candidate

Base: c09d3e20b9c45c80b34ae3781440d90904a44023.
This snapshot contains the QFM-only framework candidate. Native packages are
loaded separately; installation does not run NPU/model tests or rebuild kernels.

`source/` is the final readable candidate, not a stack of incremental patches.
`va030-qfm-only.patch` applies to the clean base. The guarded installer also
recognizes all previously delivered complete QFM/FIA/QFA/paged states, backs up
their files, removes old experiment modules/tests, and restores experiment-only
upstream hooks. Unknown or mixed changes are refused without force/reset.

Restart Workers and discard old live KV/APC pages after migration. This candidate
requires machine validation; CPU migration and unit checks are not acceptance.
