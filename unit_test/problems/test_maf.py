from unittest import TestCase

import torch

from evox.problems.numerical import (
    MAF1,
    MAF2,
    MAF3,
    MAF4,
    MAF5,
    MAF6,
    MAF7,
    MAF8,
    MAF9,
    MAF10,
    MAF11,
    MAF12,
    MAF13,
    MAF14,
    MAF15,
)


class TestMAF(TestCase):
    def setUp(self):
        d = 12
        m = 2
        self.pro = [
                MAF1(d,m),
                MAF2(d,m),
                MAF3(d,m),
                MAF4(d,m),
                MAF5(d,m),
                MAF6(d,m),
                MAF7(d,m),
                MAF8(2,3), #MAF 8 is only defined for D = 2 and M >= 3
                MAF9(2,3), #MAF 9 is only defined for D = 2 and M >= 3
                MAF10(d,m),
                MAF11(d,m),
                MAF12(d,m),
                MAF13(d,3), #MAF 13 is only defined for M >= 3
                MAF14(d,m),
                MAF15(d,m),
        ]

    def test_maf(self):
        for pro in self.pro:
            pop = torch.rand(7, pro.d)
            print(f"pro: {pro}")

            fit = pro.evaluate(pop)
            print(f"fit.size(): {fit.size()}")
            assert fit.size() == (7, pro.m)

            pf = pro.pf()
            print(f"pf.size(): {pf.size()}")
            assert pf.size(1) == pro.m
