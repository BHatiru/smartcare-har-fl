# HAR/FL results brief

Updated UTC: 2026-10-24T18:47:27.003956+00:00

Frozen target: 50 rounds; seeds [42, 43, 44]; centralized, local, FedAvg, FedBN, FedProx.
Model, graph and preprocessing unchanged. Float32 CUDA, TF32 disabled. No pooled BN calibration or warm start.
Selection uses pooled development macro-F1, then lower cross-entropy, then earliest round.

Outcome: completed with independent artifact audit (final_audit.json).

| Method | Test accuracy mean | Across-seed sample SD | Test macro-F1 mean | Across-seed sample SD |
|---|---:|---:|---:|---:|
| centralized | 0.920162 | 0.005509 | 0.920293 | 0.005541 |
| local | 0.826263 | 0.003866 | 0.826600 | 0.003432 |
| fedavg | 0.784404 | 0.015179 | 0.777817 | 0.018652 |
| fedbn | 0.834747 | 0.008517 | 0.833130 | 0.010957 |
| fedprox | 0.779960 | 0.028376 | 0.772280 | 0.032068 |

In this fixed comparison, centralized has the highest mean performance. FedBN exceeds FedAvg in pooled accuracy and macro-F1 for each paired seed; FedProx has lower means than FedAvg, with paired differences changing sign across seeds. These are descriptive benchmark results.

Producing artifacts: final_test/*.npz and *.json; independently reconstructed aggregates and paired seed differences: final_audit.json.
Per-class support/precision/recall, falling counts, client and subject summaries are in each final_test/seed_*_METHOD.json.
Selected and final states, per-round development curves and optimizer-step accounting: primary/seed_*/METHOD/.
The fixed round budget is not a claim of convergence. Across-seed SD is descriptive; no significance or general fairness claim.

Limitations: synthetic label-skew cohorts use ground-truth labels for benchmark allocation, not an available deployment router. The NTU benchmark was historically explored. Custom directed graph retained. No novel clustered method, physiological validity, synchronized fusion, privacy, deployment throughput or safety claim.

Return archive excludes the original raw/prepared tensor files and local Python environment. Keep the original bundle for reproduction.


Additional independent review: reporting/final_metric_audit.json reconstructs every pooled, per-class, client, subject and falling metric directly from saved test predictions. Identity/confusion checks are exact; independently recomputed cross-entropy uses a declared 1e-5 absolute tolerance. Selected model hashes match the frozen manifest.

Review figures: reporting/development_learning_curves.pdf, reporting/final_test_seed_results.pdf and reporting/measured_runtime.pdf (PNG copies also supplied). reporting/review.json binds plot inputs and lists every selected round and development endpoint.

Architecture: the unchanged custom 444,507-parameter ST-GCN-style network, not a faithful upstream ST-GCN++ reproduction (gpu_preflight.json and source snapshots).

Operational recovery: root-disk pressure interrupted seed 42 centralized training after round 25. The complete workspace was hash-verified on the user-unlocked volume, dependencies matched exactly, and all selected development logits replayed exactly before resuming. Recovery evidence is preserved in storage_recovery/. Matched examples and optimizer steps describe the retained completed-round trajectory; the exact discarded partial-round step count was not logged, and reported training seconds exclude that discarded work and the recovery pause. See RESUME_ACCOUNTING.md.
