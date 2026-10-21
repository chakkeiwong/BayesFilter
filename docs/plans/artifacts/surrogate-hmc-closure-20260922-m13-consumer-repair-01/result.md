# M13 observability repair — bounded CPU regression

The full expanded correction diagnostic payload is now opt-in. The default
claim-bearing batch consumer retains its previous five scalar cap diagnostics,
because changing an XLA output signature changed FP32 operation ordering in the
prior M13 parity attempt. The authority regression requests
return_all_correction_diagnostics=True and compares every returned field with
the single-cloud correction authority.

The first focused run after the API change exposed a shape mismatch in the
restored default fields; the implementation was corrected to scalar fields.
The final bounded CPU suite passed 38 tests. This repair consumed no GPU or
numerical M13 attempt. It does not qualify M13 and does not erase the prior
GPU no-harm failure. A new GPU no-harm retest belongs to a fresh scoped plan.
