from pathlib import Path
import sys
import unittest


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 将项目根目录加入模块搜索路径
sys.path.insert(0, str(PROJECT_ROOT))

from src.class_parser import parse_classes


class TestClassParser(unittest.TestCase):

    def test_single_class(self):
        """单个类别。"""
        self.assertEqual(
            parse_classes("person"),
            ["person"]
        )

    def test_multiple_phrases(self):
        """多个类别，并保留短语内的空格。"""
        self.assertEqual(
            parse_classes("person, traffic light, fire extinguisher"),
            ["person", "traffic light", "fire extinguisher"]
        )

    def test_normalize_whitespace(self):
        """清理首尾和内部多余空白。"""
        self.assertEqual(
            parse_classes("  person  , traffic   light, fire\t extinguisher "),
            ["person", "traffic light", "fire extinguisher"]
        )

    def test_ignore_empty_items(self):
        """忽略分隔符产生的空项。"""
        self.assertEqual(
            parse_classes(", person, , bus, "),
            ["person", "bus"]
        )

    def test_remove_duplicates(self):
        """忽略大小写去重，保留首次写法和顺序。"""
        self.assertEqual(
            parse_classes("Person, bus, person, BUS, traffic light"),
            ["Person", "bus", "traffic light"]
        )

    def test_chinese_comma(self):
        """兼容中文逗号。"""
        self.assertEqual(
            parse_classes("person，traffic light"),
            ["person", "traffic light"]
        )

    def test_no_fixed_whitelist(self):
        """自由类别短语不会被固定词表过滤。"""
        self.assertEqual(
            parse_classes("electric scooter, red toolbox, striped umbrella"),
            ["electric scooter", "red toolbox", "striped umbrella"]
        )

    def test_empty_input(self):
        """没有有效类别时报告错误。"""
        for text in ("", "   ", ",,,", " ， , "):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_classes(text)

    def test_non_string_input(self):
        """非字符串输入报告类型错误。"""
        for value in (None, 123, ["person"], {}):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    parse_classes(value)

    def test_chinese_input(self):
        """中文或中英文混合输入报告错误。"""
        for text in ("人", "person, 汽车", "交通灯，bus"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_classes(text)


if __name__ == "__main__":
    unittest.main(verbosity=2)