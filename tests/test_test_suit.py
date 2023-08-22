from evox import problems

def test_cec2022_test_suit():
    for i in range(1, 13, 1):
        pro = problems.test_suit.CEC2022.create(i)
        pro.init()