from __future__ import annotations

import unittest
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from hakyking.controllers.main_controller import MainController


class RenderResultCacheTests(unittest.TestCase):
    def test_cache_is_lru_and_respects_memory_limit(self) -> None:
        controller = MainController.__new__(MainController)
        controller._render_result_cache = OrderedDict()
        controller._render_result_cache_bytes = 0
        controller.RENDER_RESULT_CACHE_MAX_BYTES = 24
        controller.RENDER_RESULT_CACHE_MAX_ITEM_BYTES = 16

        first = SimpleNamespace(cache_key="first", audio=np.zeros(4, dtype=np.float32))
        second = SimpleNamespace(cache_key="second", audio=np.zeros(4, dtype=np.float32))
        controller._cache_render_result(first)
        controller._cache_render_result(second)

        self.assertNotIn("first", controller._render_result_cache)
        self.assertIn("second", controller._render_result_cache)
        self.assertEqual(controller._render_result_cache_bytes, 16)

    def test_oversized_result_is_not_cached(self) -> None:
        controller = MainController.__new__(MainController)
        controller._render_result_cache = OrderedDict()
        controller._render_result_cache_bytes = 0
        controller.RENDER_RESULT_CACHE_MAX_BYTES = 64
        controller.RENDER_RESULT_CACHE_MAX_ITEM_BYTES = 8
        result = SimpleNamespace(cache_key="large", audio=np.zeros(4, dtype=np.float32))

        controller._cache_render_result(result)

        self.assertFalse(controller._render_result_cache)
        self.assertEqual(controller._render_result_cache_bytes, 0)


if __name__ == "__main__":
    unittest.main()
