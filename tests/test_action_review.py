import copy
import unittest

import numpy as np

from gesture.action_annotations import ActionAnnotationEditor
from gesture.action_candidates import action_excluded_frames, action_opportunities, propose_action_interval
from gesture.action_data import make_action_targets
from gesture.action_evaluation import action_event_metrics
from gesture.action_features import ACTION_FEATURE_SIZE
from gesture.collection import alternating_action_prompts, collection_phases
from gesture.config import load_config


def session_fixture():
    prompts = [{"gesture": "close_hand", "start_frame": 20, "end_frame": 49},
               {"gesture": "open_hand", "start_frame": 70, "end_frame": 99}]
    features = np.zeros((120, ACTION_FEATURE_SIZE), np.float32)
    for frame in range(120):
        closing = min(1.0, max(0.0, (frame - 20) / 9))
        if frame >= 70:
            closing = 1 - min(1.0, (frame - 70) / 9)
        features[frame, :42] = closing * 0.3
        features[frame, -2:] = (0.9 * (1 - closing), 0.9 * closing)
    return {"features": features, "times": np.arange(120) / 10,
            "valid": np.ones(120, bool), "segments": np.zeros(120, int),
            "opportunities": action_opportunities(prompts, 120)}


class ActionReviewTest(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.session = session_fixture()
        self.annotations = {"action_intervals": [], "action_candidates": [], "action_ignored_intervals": [],
                            "action_reviewed": False, "action_review_mode": "prompts"}

    def editor(self):
        return ActionAnnotationEditor(self.annotations, 120, self.session["valid"], ["open_hand", "close_hand"], self.session, self.config)

    def test_alternating_prompts_and_hold_pose_never_require_reset(self):
        prompts = alternating_action_prompts(3, "opened")
        self.assertEqual([item["gesture"] for item in prompts], ["close_hand", "open_hand"] * 3)
        for prompt in prompts:
            prompt.update(duration_seconds=3, hold_seconds=2)
        plan = {"target_model": "action", "initial_pose": "opened", "prepare_seconds": 2,
                "sample_type": "positive", "prompts": prompts}
        phases = collection_phases(plan)
        self.assertEqual(phases[0]["gesture"], "opened")
        for index, prompt in enumerate(prompts):
            hold = phases[index * 2 + 2]
            self.assertEqual(hold["phase"], "hold")
            self.assertEqual(hold["gesture"], "closed" if prompt["gesture"] == "close_hand" else "opened")
        self.assertEqual(phases[-1]["end"], 32)

    def test_candidates_are_drafts_and_no_source_pose_produces_no_label(self):
        draft = propose_action_interval(self.session, self.session["opportunities"][0], self.config)
        self.assertIsNotNone(draft["start_frame"])
        self.assertLess(draft["start_frame"], draft["end_frame"])
        self.assertEqual(draft["label"], "close_hand")
        editor = self.editor()
        self.assertEqual(editor.intervals, [])
        self.assertEqual(len(self.annotations["action_candidates"]), 2)
        absent = copy.deepcopy(self.session)
        absent["features"][:, -2:] = [0.0, 0.9]
        draft = propose_action_interval(absent, absent["opportunities"][0], self.config)
        self.assertIsNone(draft["start_frame"])
        self.assertIsNone(draft["end_frame"])

    def test_confirm_then_cancel_auto_advance_and_mask_whole_opportunity(self):
        editor = self.editor()
        editor.apply("confirm", editor.focus_frame())
        self.assertEqual(editor.prompt_position, 1)
        self.assertEqual(len(editor.intervals), 1)
        self.assertFalse(self.annotations["action_reviewed"])
        editor.apply("cancel", editor.focus_frame())
        self.assertTrue(self.annotations["action_reviewed"])
        excluded = action_excluded_frames(self.annotations, self.session["opportunities"], 120, self.config)
        targets, mask = make_action_targets(self.session["valid"], self.session["segments"], editor.intervals,
                                           True, self.config, excluded)
        self.assertFalse(mask[55:].any())
        self.assertTrue((targets[:, 1] * mask[:, 1]).any())
        self.assertFalse(targets[:, 0].any())
        self.assertTrue(excluded[100:].all())  # The cancelled HOLD is excluded too.

    def test_pending_is_ignored_even_when_old_review_flag_is_true(self):
        self.editor()
        excluded = action_excluded_frames(self.annotations, self.session["opportunities"], 120, self.config)
        _, mask = make_action_targets(self.session["valid"], self.session["segments"], [], True, self.config, excluded)
        self.assertFalse(mask[20:].any())

    def test_adjustment_requires_confirmation_and_survives_reopen(self):
        editor = self.editor()
        editor.apply("confirm", editor.focus_frame())
        editor.apply("previous_prompt", 0)
        editor.apply("start", 23)
        self.assertEqual(editor.status(), "PENDING")
        self.assertEqual(editor.intervals, [])
        reopened = self.editor()
        self.assertEqual(reopened.current_interval()["start_frame"], 23)
        reopened.apply("confirm", 23)
        self.assertEqual(reopened.intervals[0]["start_frame"], 23)

    def test_cancelled_predictions_do_not_count_as_false_positives(self):
        editor = self.editor()
        editor.apply("confirm", editor.focus_frame())
        editor.apply("cancel", editor.focus_frame())
        session = {**self.session, "intervals": editor.intervals,
                   "ignored_frames": action_excluded_frames(self.annotations, self.session["opportunities"], 120, self.config)}
        predictions = np.zeros((120, 2), np.float32)
        predictions[90:95, 0] = 0.99
        metrics = action_event_metrics(session, predictions, self.config)
        self.assertEqual(metrics["per_class"]["open_hand"]["false_positive"], 0)

    def test_state_collection_still_has_rest_and_original_prompt_count(self):
        plan = {"target_model": "state", "gap_seconds": 2, "durations_seconds": {"opened": 3, "closed": 3},
                "prompts": [{"gesture": "opened"}, {"gesture": "closed"}]}
        phases = collection_phases(plan)
        self.assertEqual([phase["phase"] for phase in phases], ["state", "rest", "state"])
        self.assertEqual(phases[-1]["end"], 8)


class NegativeActionReviewTest(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.prompts = [
            {"gesture": "hold_opened", "sample_type": "negative", "start_frame": 20, "end_frame": 59},
            {"gesture": "hold_closed", "sample_type": "negative", "start_frame": 80, "end_frame": 119},
        ]
        self.valid = np.ones(120, bool)
        self.valid[50:53] = False
        segments = np.zeros(120, int)
        segments[52:] = 1
        self.session = {
            "features": np.zeros((120, ACTION_FEATURE_SIZE), np.float32),
            "times": np.arange(120) / 10,
            "valid": self.valid,
            "segments": segments,
            "opportunities": action_opportunities(self.prompts, 120),
        }
        self.annotations = {
            "action_intervals": [], "action_candidates": [], "action_ignored_intervals": [],
            "action_background_prompts": [], "action_reviewed": False, "action_review_mode": "prompts",
        }

    def editor(self):
        return ActionAnnotationEditor(self.annotations, 120, self.valid, ["open_hand", "close_hand"], self.session, self.config)

    def targets(self):
        excluded = action_excluded_frames(self.annotations, self.session["opportunities"], 120, self.config)
        return make_action_targets(self.valid, self.session["segments"], self.annotations["action_intervals"],
                                   self.annotations["action_reviewed"], self.config, excluded)

    def test_negative_does_not_generate_event_truth_and_requires_background_confirmation(self):
        editor = self.editor()
        self.assertEqual(editor.status(), "PENDING")
        self.assertEqual(self.annotations["action_candidates"], [])
        self.assertEqual(editor.intervals, [])
        _, mask = self.targets()
        self.assertFalse(mask.any())
        editor.apply("confirm", 20)
        self.assertEqual(editor.status(0), "BACKGROUND")
        self.assertEqual(self.annotations["action_background_prompts"], [0])
        self.assertEqual(editor.prompt_position, 1)
        self.assertFalse(self.annotations["action_reviewed"])
        editor.apply("confirm", 80)
        self.assertTrue(self.annotations["action_reviewed"])
        target, mask = self.targets()
        self.assertFalse(target.any())
        self.assertTrue(mask[25:40].all())
        self.assertTrue(mask[85:100].all())
        self.assertFalse(mask[:20].any())
        self.assertFalse(mask[60:80].any())
        self.assertFalse(mask[50:53].any())
        self.assertFalse(mask[53:55].any())  # Background waits for consecutive visible frames.

    def test_negative_opportunities_exclude_prepare_without_consuming_next_task(self):
        opportunities = self.session["opportunities"]
        self.assertEqual([(item["start_frame"], item["end_frame"]) for item in opportunities], [(20, 59), (80, 119)])
        editor = self.editor()
        editor.apply("confirm", 20)
        editor.apply("cancel", 80)
        self.assertEqual(editor.status(1), "IGNORE")
        self.assertTrue(self.annotations["action_reviewed"])
        _, mask = self.targets()
        self.assertTrue(mask[25:40].all())
        self.assertFalse(mask[60:].any())

    def test_real_events_in_negative_task_can_be_saved_deleted_and_reconfirmed(self):
        editor = self.editor()
        editor.apply("confirm", 20)
        editor.apply("confirm", 80)
        editor.apply("previous_prompt", 80)
        self.assertEqual(editor.status(), "BACKGROUND")
        for label, begin, finish in (("label_0", 24, 30), ("label_1", 40, 44)):
            editor.apply(label, begin)
            editor.apply("start", begin)
            self.assertFalse(self.annotations["action_reviewed"])
            self.assertEqual(editor.status(), "PENDING")
            editor.apply("finish", finish)
            editor.apply("add_event", finish)
        self.assertEqual([(item["label"], item["start_frame"], item["end_frame"]) for item in editor.intervals],
                         [("open_hand", 24, 30), ("close_hand", 40, 44)])
        self.assertEqual([item["prompt_index"] for item in editor.intervals], [0, 0])
        reopened = self.editor()
        self.assertEqual(reopened.intervals, editor.intervals)
        self.assertEqual(reopened.status(), "PENDING")
        reopened.apply("delete_event", 26)
        self.assertEqual(len(reopened.intervals), 1)
        self.assertEqual(reopened.intervals[0]["label"], "close_hand")
        reopened.apply("confirm", 20)
        self.assertTrue(self.annotations["action_reviewed"])
        target, mask = self.targets()
        np.testing.assert_array_equal(np.flatnonzero(target[:, 1]), np.arange(47, 52))
        self.assertFalse(target[:, 0].any())
        self.assertTrue(mask[47:50, 1].all())
        self.assertFalse(mask[50:53].any())


if __name__ == "__main__":
    unittest.main()
