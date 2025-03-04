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
                MAF8(2,3),
                MAF9(2,3),
                MAF10(d,m),
                MAF11(d,m),
                MAF12(d,m),
                MAF13(d,3),
                MAF14(d,m),
                MAF15(d,m),
        ]

    def test_maf(self):
        for pro in self.pro:
            if pro.__class__ in [MAF11]:
                pop = torch.rand(7, 11)
                fit = pro.evaluate(pop)
                print(pro.__class__.__name__)
                if not fit.size() == (7, 2):
                    print(fit.size())
                pf = pro.pf()
                if pf.size(1) == 2:
                    print(pf.size())
            elif pro.__class__ in [MAF8, MAF9, MAF13]:
                pop = torch.rand(7, 12)
                fit = pro.evaluate(pop)
                print(pro.__class__.__name__)
                if not fit.size() == (7, 3):
                    print(fit.size())
                pf = pro.pf()
                if pf.size(1) == 3:
                    print(pf.size())
            else:
                pop = torch.rand(7, 12)
                fit = pro.evaluate(pop)
                print(pro.__class__.__name__)
                if not fit.size() == (7, 2):
                    print(fit.size())
                pf = pro.pf()
                if pf.size(1) == 2:
                    print(pf.size())

if __name__ == "__main__":
    test = TestMAF()
    test.setUp()
    test.test_maf()
