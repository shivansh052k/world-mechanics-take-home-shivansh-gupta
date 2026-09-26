# Talk outline (~15 min, basis for discussion)

Status markers: [done] results exist · [todo] not yet run.

1. **Question and setup** (1.5 min) [done]
   Frozen V-JEPA 2 ViT-L/16; one disk moving on a flat background; what does the model encode about direction,
   speed and acceleration, and where?
2. **Data and protocol** (2 min) [done]
   What the audit found (orange disk, exact geometry, sorted manifests, exit clips, distance confound); splits with
   held-out values; how activations are extracted and pooled; test used once.
3. **Layer-wise probing** (2.5 min) [done] — slide "Where motion becomes readable across layers"
4. **Local to global, direction** (2 min) [done] — slide "Where inside the frame direction is readable"
5. **Nullspace probing: what the count measures** (2 min) [done] — slides "What the nullspace count measures",
   "Linear erasure removes the readout, not the information"
6. **Multi-probe subspace steering on held-out clips and values** (2 min) [done] — slides "Steering works at the
   steering layer, not nine blocks later", "Where the edit is lost"
7. **Spline steering and the comparison with probe steering** (2 min) [todo]
8. **Confounds and robustness** (1 min) [todo] — distance vs acceleration, motion type, flags
9. **Limitations and open questions** (1 min) [todo]
   So far: time-averaged per-patch vectors; simpler stimuli than the paper; one checkpoint; interpretation of unspecified protocol details; nullspace counts depend on the probe's regularization; only linear erasure and one kernel family tested; speed / acceleration labels equal distance travelled. same-layer steering evaluation follows by construction; uniform token edit; one steering / readout layer.