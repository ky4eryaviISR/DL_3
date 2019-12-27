import random

try:
    from xeger import Xeger
except ImportError:
    import subprocess
    import os
    import sys

    cfolder = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-m", "pip", "install", "--target=" + cfolder, 'xeger']
    subprocess.call(cmd)
    from xeger import Xeger



def is_prime(n):
    if n==2 or n==3:
        return True
    if n%2==0 or n<2:
        return False
    for i in range(3,int(n**0.5)+1,2):   # only odd numbers
        if n%i==0:
            return False

    return True

def generate_prime():
    good = []
    bad = []
    count = 0
    while count < 500:
        i = random.randint(2, 99999)
        i = i if i % 2 == 1 else i+1
        if i in good or i in bad:
            continue
        if is_prime(i):
            print(i)
            good.append(str(i))
            count += 1
        elif len(bad) < 500:
            bad.append(str(i))
        i += 1

    with open('pos_examples_prime', 'w') as f:
        f.write('\n'.join(good))
    with open('neg_examples_prime', 'w') as f:
        f.write('\n'.join(bad))


def generator():
    good = []
    bad = []
    for i in range(500):
        while True:
            g = random.randint(10, 100000)
            g = g*9
            if g not in good:
                break
        good.append(str(g))
        while True:
            b = random.randint(10, 100000)
            if b % 9 == 0:
                b += 1
            if b not in bad:
                break
        bad.append(str(b))


    with open('pos_examples_generator', 'w') as f:
        f.write('\n'.join(good))
    with open('neg_examples_generator', 'w') as f:
        f.write('\n'.join(bad))


#generate_prime()
generator()
