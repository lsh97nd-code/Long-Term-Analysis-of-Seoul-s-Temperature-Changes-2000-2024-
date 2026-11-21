"""기존 분석 코드의 시계열 분해 기능을 실제 CSV로 검증합니다."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import analysis as an


class SeoulDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = an.load_raw()
        cls.df, cls.log = an.clean(cls.raw)
        cls.dec = an.decompose(cls.df)

    def test_daily_record_count(self):
        self.assertEqual(len(self.raw), 9132)
        self.assertEqual(str(self.raw['date'].min().date()), '2000-01-01')
        self.assertEqual(str(self.raw['date'].max().date()), '2024-12-31')

    def test_daily_index_has_no_gaps(self):
        self.assertEqual(int(self.log['duplicates_removed']), 0)
        self.assertEqual(int(self.log['dates_added']), 0)

    def test_four_components_exist(self):
        self.assertEqual(list(self.dec.columns), ['observed', 'trend', 'seasonal', 'resid'])
        self.assertEqual(len(self.dec), 9132)

    def test_additive_identity(self):
        valid = self.dec.dropna(subset=['observed', 'trend', 'seasonal', 'resid'])
        self.assertGreater(len(valid), 8000)
        recomposed = valid['trend'] + valid['seasonal'] + valid['resid']
        self.assertTrue(np.allclose(valid['observed'].to_numpy(), recomposed.to_numpy(), atol=1e-9))

    def test_seasonality_is_not_all_zero(self):
        amp = self.dec['seasonal'].max() - self.dec['seasonal'].min()
        self.assertGreater(amp, 20)
        self.assertLess(amp, 40)

    def test_decomposition_png_exists(self):
        png = ROOT / 'images' / '06_decomposition.png'
        self.assertTrue(png.exists())
        self.assertGreater(png.stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main()
