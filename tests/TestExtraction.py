from src.extraction import extract_text_from_file

import unittest


class TestExtraction(unittest.TestCase):
    def test_extract_text_from_file(self):
        test = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        extract = extract_text_from_file()
        for line in extract:
            for turples in line:
                if len(turples) == 3:
                    test[turples[2]] = test[turples[2]] + 1

        self.assertEqual(test[1], 361)
        self.assertEqual(test[2], 11)
        self.assertEqual(test[3], 66)
        self.assertEqual(test[4], 178)
        self.assertEqual(test[5], 500)
        self.assertEqual(test[6], 1252)
