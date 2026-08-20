import unittest

from simulator.parser import find_simulation_errors, parse_measurements


class ParserTests(unittest.TestCase):
    def test_ignores_unexpected_numeric_assignments_when_schema_is_given(self):
        parsed = parse_measurements("wanted = 1\nnoise = 2", ("wanted",))
        self.assertEqual(parsed.measurements, {"wanted": 1.0})

    def test_parses_scientific_measurements_case_insensitively(self):
        output = "VOUT = 2.500000e-01\n gain_db = -6.0206\n"
        result = parse_measurements(output)
        self.assertEqual(result.measurements, {"vout": 0.25, "gain_db": -6.0206})
        self.assertEqual(result.errors, ())

    def test_rejects_duplicate_measurements(self):
        result = parse_measurements("vout = 0.25\nvout = 0.26\n", ("vout",))
        self.assertEqual(result.measurements, {"vout": 0.25})
        self.assertIn("Duplicate measurement", result.errors[0])

    def test_rejects_nonfinite_measurement(self):
        result = parse_measurements("vout = NaN\n", ("vout",))
        self.assertEqual(result.measurements, {})
        self.assertIn("non-finite", result.errors[0])

    def test_rejects_malformed_expected_measurement(self):
        result = parse_measurements("vout = not-a-number\n", ("vout",))
        self.assertEqual(result.measurements, {})
        self.assertIn("invalid", result.errors[0])

    def test_detects_convergence_and_measurement_failures(self):
        output = "Warning only\nError: singular matrix\nMeasurement vout failed!\n"
        errors = find_simulation_errors(output)
        self.assertEqual(len(errors), 2)
        self.assertEqual(len(find_simulation_errors(output)), 2)


if __name__ == "__main__":
    unittest.main()
