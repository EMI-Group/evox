from unittest import TestCase

import torch

from evox.problems.numerical import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7


class TestBraxProblem(TestCase):
    def setUp(self):
        d = 12
        m = 3
        self.pro = [
            DTLZ1(d=d, m=m),
            DTLZ2(d=d, m=m),
            DTLZ3(d=d, m=m),
            DTLZ4(d=d, m=m),
            DTLZ5(d=d, m=m),
            DTLZ6(d=d, m=m),
            DTLZ7(d=d, m=m),
        ]

    def test_dtlz(self):
        pop = torch.rand(2, 12)
        for pro in self.pro:
            fit = pro.evaluate(pop)
            assert fit.size() == (2, 3)
            pf = pro.pf()
            assert pf.size(1) == 3
