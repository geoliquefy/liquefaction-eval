"""Mirror-only test settings. This file is not in the kit repo.

The tests below read private driver scripts or a private data-source list that
this read-only mirror does not hold. They are skipped here, by name, with the
reason. Every other kit test file is byte-identical to kit commit f77667a.
"""
import pytest

_SCRIPT = "reads a private driver script that this mirror does not hold"
_SKIP = {
    "tests/test_evalkit_d8_elbistan_pga.py::test_driver_elbistan_step_names_no_label_path": _SCRIPT,
    "tests/test_evalkit_d8_elbistan_pga.py::test_driver_elbistan_step_end_to_end_on_a_synthetic_api": _SCRIPT,
    "tests/test_evalkit_d8_zhu_refreeze.py::test_driver_refreeze_step_names_no_label_path": _SCRIPT,
    "tests/test_evalkit_d8_zhu_refreeze.py::test_driver_stamp_plan_names_the_v2_manifest_first": _SCRIPT,
    "tests/test_evalkit_world_inputs.py::test_driver_predict_step_names_no_label_path": _SCRIPT,
    "tests/test_evalkit_zhu_grid.py::test_d8_predict_passes_the_row_order_to_the_zhu_read": _SCRIPT,
    "tests/test_evalkit_zhu_grid.py::test_d8_predict_row_order_choices_are_the_reader_names": _SCRIPT,
    "tests/test_evalkit_sequences.py::test_p_max_matches_the_old_two_shock_script_exactly": _SCRIPT,
    "tests/test_evalkit_sequences.py::test_legacy_two_shock_map_is_the_old_fore_map": _SCRIPT,
    "tests/test_evalkit_parity_round3.py::test_table_counts_equal_the_g1_curated_readme_counts": _SCRIPT,
    "tests/test_evalkit_parity_round3.py::test_runner_refuses_without_go_before_opening_any_file": _SCRIPT,
    "tests/test_evalkit_parity_round3.py::test_runner_chain_path_on_planted_profiles": _SCRIPT,
    "tests/test_evalkit_parity_round3.py::test_runner_word_uses_ties_half_and_the_published_string": _SCRIPT,
    "tests/test_evalkit_parity_round3.py::test_runner_stops_with_no_word_on_a_planted_row_fault": _SCRIPT,
    "tests/test_evalkit_parity_round3.py::test_runner_beside_fault_keeps_the_word": _SCRIPT,
    "tests/test_evalkit_parity_round3.py::test_runner_published_constant_matches_its_header_and_the_registration": _SCRIPT,
    "tests/test_evalkit_d8_predictions_v3.py::test_driver_predict_v3_step_names_no_label_path": _SCRIPT,
    "tests/test_evalkit_d8_predictions_v3.py::test_driver_stamp_plan_names_v3_first": _SCRIPT,
    "tests/test_evalkit_d8_score.py::test_changed_label_sha_stops_before_any_parse": _SCRIPT,
    "tests/test_evalkit_d8_score.py::test_existing_output_with_a_metric_refuses_the_run": _SCRIPT,
    "tests/test_evalkit_d8_score.py::test_kit_loads_from_this_worktree": _SCRIPT,
    "tests/test_evalkit_d8_score.py::test_module_has_no_em_dashes": _SCRIPT,
    "tests/test_evalkit_graderkit.py::test_pillar_sources_carry_notice_quality_and_dec_flag":
        "reads a private data-source list that this mirror does not hold",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = item.nodeid.split("[", 1)[0]
        if name in _SKIP:
            item.add_marker(pytest.mark.skip(reason=_SKIP[name]))
