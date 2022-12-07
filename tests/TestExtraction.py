from src.extraction import extract_text_from_file, create_word_package, create_syntax_package

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

    def test_create_word_package(self):

        extract = extract_text_from_file()
        test = create_word_package(extract, 2, 2)

        self.assertEqual(len(test[0]), 2368)
        self.assertEqual(len(test[1]), 2368)
        self.assertGreater(len(test[0][0]), 1)

    def test_create_syntax_package(self):

        extract = extract_text_from_file()
        test = create_syntax_package(extract, 2, 2)

        self.assertEqual(len(test[0]), 2368)
        self.assertEqual(len(test[1]), 2368)
        self.assertGreater(len(test[0][0]), 1)
