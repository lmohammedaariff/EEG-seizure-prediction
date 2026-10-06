import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labeling.exclusion import Seizure, label_window


class ExclusionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.seizure = Seizure(600, 660, "chb01_03.edf:sz1", "chb01_03.edf")

    def test_preictal_candidate_inside_horizon(self):
        result = label_window(300, 304, [self.seizure], 5)
        self.assertEqual(result.label, 1)
        self.assertEqual(result.seizure_id, self.seizure.seizure_id)

    def test_window_crossing_preictal_start_is_excluded(self):
        result = label_window(298, 302, [self.seizure], 5)
        self.assertEqual(result.label, -1)
        self.assertIn("exclusion", result.reason)

    def test_window_crossing_seizure_onset_is_excluded(self):
        result = label_window(598, 602, [self.seizure], 5)
        self.assertEqual(result.label, -1)

    def test_fully_ictal_window(self):
        result = label_window(604, 608, [self.seizure], 5)
        self.assertEqual(result.label, 2)

    def test_postictal_buffer_is_excluded_then_interictal_resumes(self):
        excluded = label_window(660, 664, [self.seizure], 5, postictal_exclusion_seconds=60)
        ordinary = label_window(720, 724, [self.seizure], 5, postictal_exclusion_seconds=60)
        self.assertEqual(excluded.label, -1)
        self.assertEqual(ordinary.label, 0)

    def test_interictal_exclusion_buffer_is_explicit(self):
        result = label_window(200, 204, [self.seizure], 5, interictal_exclusion_seconds=120)
        self.assertEqual(result.label, -1)
        self.assertIn("interictal-to-preictal", result.reason)

    def test_patient_clock_can_label_a_window_in_an_earlier_recording(self):
        global_event = Seizure(3700, 3760, "next.edf:sz1", "next.edf")
        result = label_window(1900, 1904, [global_event], 30)
        self.assertEqual(result.label, 1)
        self.assertEqual(result.seizure_recording_id, "next.edf")

    def test_unsupported_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            label_window(300, 304, [self.seizure], 20)


if __name__ == "__main__":
    unittest.main()
