# QSFA full-request baseline (2026-10-31)

Immutable installer commit: `14fd9f5311a33e7ba1677402281a6589d0c6e936`, file `install-c8-full-request.py`.
Requires the existing VA 0.29 K3 C8 + DSpark adaptation; not a clean VA installation and not VA 0.30.

Both eager readers submit each actual nonempty request once (no fixed 8192 and no automatic index-budget split). FP8 latent512 + BF16 raw64 + FP32 group128 scales, packed656. Decode graph unchanged. No external historical KV dequantization.

User observed approximately 14 minutes -> 4 minutes for a long request after removing internal query splitting, still slower than their BF16 comparison. These are user observations, not controlled benchmark results. CPU 274 tests passed; full-request native shapes/performance and DCP whole-model acceptance not certified.

The new FIA branch is a separate experiment with explicit external dequantization authorization; it does not supersede or establish acceptance for this baseline.
