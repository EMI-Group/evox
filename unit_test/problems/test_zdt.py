from unittest import TestCase

import torch

from evox.problems.numerical import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6


class TestZDT(TestCase):

    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.n = 12
    def setUp(self):
        self.pro = [
            ZDT1(n=self.n),
            ZDT2(n=self.n),
            ZDT3(n=self.n),
            ZDT4(n=self.n),
            ZDT6(n=self.n),
        ]

    def test_zdt(self):
        pop = torch.rand(7, self.n)
        for pro in self.pro:
            print(f"pro: {pro}")
            fit = pro.evaluate(pop)
            print(f"fit.size(): {fit.size()}")
            assert fit.size() == (7, 2)
            pf = pro.pf()
            print(f"pf.size(): {pf.size()}")
            assert pf.size(1) == 2
