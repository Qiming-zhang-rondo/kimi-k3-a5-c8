# Invalid timing entries retained for audit

The six `.snapshot` files preserve the exact Git blobs from
`5407a8e8175f750f859e061a5b993815a8756477`; SHA256, byte counts and Git blob SHA1
are recorded in `manifest.json`. They are evidence, never build or execution inputs.
Their original paths now fail explicitly. `SFA_BUILD_PHASE_C=ON` also fails before
toolkit discovery, so an old command cannot continue this benchmark accidentally.

The disabled implementations used zero inputs/invalid present cold states,
changed the validated compact128 geometry, omitted output correctness gates,
ignored ACL failures and reset device 0. Some included both incompatible kernel
definitions in one TU; `sfa_fusion_entry_main.asc` exported an empty kernel.
Historical 87.53/126.17 microsecond constants were not paired measurements.
These files establish no C4 W0, two-pass, three-arm or performance result.

The replacement build is `-DSFA_BUILD_TORCH_DIRECT=ON`: it produces
`libsfa_torch_direct.so` from two separate ASC wrapper translation units.
`sfa_direct_api.h` contains only a C-layout pointer struct and host declarations;
the Torch binding owns tensor validation and passes its actual current stream.
The wrappers do not initialize, synchronize, select or reset a device. They launch
the preserved original and candidate entries, using the fixed compact128 contract.
Their C ABI version is 1; the buffer descriptor contains exactly 19 pointers.

The direct library is an integration component, not a standalone measurement.
The gated runner must check the same active SDK/device, independent math (0.01),
actual native-hot plus given-FP32-cold reference (0.001), finite outputs and raw
same-run ABBA timing. This narrow test does not establish a C4 end-to-end pipeline.
