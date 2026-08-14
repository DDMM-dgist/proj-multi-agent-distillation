"""Campaign-specific slice-membership adapters for the generic coverage engine.

Generic coverage code (coverage.reference_pool, coverage.nn_distance,
coverage.aggregate, coverage.report) never reads `config_type` or any other
fixed metadata field from a structure directly -- it only accepts caller-
supplied `slice_membership` / `query_slice_labels`, free-form label sequences
keyed by whatever `structure_id`s the caller chose (see
coverage/representation.py and coverage/reference_pool.py). Modules in this
package are where a specific campaign's own metadata convention is translated
into that generic shape. `coverage.adapters.sio2_config_type` is one example,
for the current SiO2-x campaign; a different material system or campaign
needs a different adapter module here, never a change to generic coverage.*
code.
"""
