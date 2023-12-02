from evox import problems

def test_cec2022_test_suit():
    for i in range(1, 13, 1):
        pro = problems.numerical.CEC2022TestSuit.create(i)
        pro.init()