# Interrupted-run accounting

Disk pressure stopped seed 42 centralized training after the round-25 checkpoint, during the following round. The saved checkpoint, development predictions, live audit, status and log are preserved under storage_recovery/. The same checkpoint will resume on the unlocked volume with unchanged model, data, seed, precision, schedule and 50-round target.

Reported matched examples and optimizer steps count the retained, completed-round training trajectory. Work in the interrupted round is discarded and deterministically replayed. The exact discarded-batch/optimizer-step count was not logged; do not describe the retained-step totals as all physical computation performed. Likewise, result training/total seconds exclude discarded work and the storage/environment recovery pause. Pipeline launch records and logs retain wall-clock provenance. Logical payload byte counts are not measured network traffic.

No test predictions were opened before this operational recovery, and no hyperparameter, checkpoint-selection rule or endpoint budget was changed based on test data.
