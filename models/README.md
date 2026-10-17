# Model artifacts

The modelling scripts create a LightGBM text model, a PyTorch state dictionary,
their fitted preprocessors, and feature-engineering artifacts in this folder.
`business_context.json` adds aggregate category comparisons and the empirical
OOF error radius used by the decision-support layer.

Generated models are intentionally ignored because they are derived from an
access-controlled competition dataset. An authorised dataset holder can rebuild
them with the commands in the project README. Cross-validation metrics are
evidence of local evaluation, not a claim that either model is deployed.
