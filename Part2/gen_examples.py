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


def generate_pattern():
    good = []
    bad = []
    x = Xeger(limit=15)
    for i in range(500):
        good.append(x.xeger(r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'))
        bad.append(x.xeger(r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+'))
    with open('pos_examples', 'w') as f:
        f.write('\n'.join(good))
    with open('neg_examples', 'w') as f:
        f.write('\n'.join(bad))


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
            good.append(str(i))
            count += 1
        elif len(bad) < 500:
            bad.append(str(i))
        i += 1

    with open('pos_examples_prime', 'w') as f:
        f.write('\n'.join(good))
    with open('neg_examples_prime', 'w') as f:
        f.write('\n'.join(bad))



def generator_divider():
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



import numpy as np

# creat polindrom-  mirrow sequnce. example ABCBA
# total size is 10000
def create_sequence(letters,neg):
    """
    Creat a polindgrom siquence, using  number-leters, each side size is maximum 100
    :param letters: A list of the letters in the sequence. Positive and negative differ from each other by this order.
    :return: A sequence of digits and letters
    """
    N = 5  # seq size (number - letters - number - letters - number - letters - number - letters - number)
    seq_str = ''
    # create list of list,where each list contain 9 numbers each number is the length of numbers or the letters duplication
    random_lens = np.random.randint(1, 10, size=N)  # random length list
    # create the sequence
    for i in range(N):  # loop over each part of the sequence (number or letter)
        # all odd places - the number
        if i % 2 == 0:
            # randomly choose a number, by joining integers [1-9] of random length.
            seq_str += ''.join([str(i) for i in np.random.randint(1, 10, size=random_lens[i])])
        else:  # choose a sequence of identical letters in a random length.
            seq_str += ''.join(letters[i // 2] * random_lens[i])

    # creat a pos or neg polindrom
    if not neg:
        rev_seq_str = seq_str[::-1]
        polindrom_seq_str = seq_str + rev_seq_str
    else:
        rev_seq_str = seq_str[::-1] # get the reverse
        i = np.random.randint(1, len(seq_str))
        ji=np.random.randint(1,len(seq_str))
        while seq_str[i]==seq_str[ji]:
            i = np.random.randint(1, len(seq_str))
            ji = np.random.randint(1, len(seq_str))
        seq_str=seq_str.replace(seq_str[i],seq_str[ji])
        polindrom_seq_str = seq_str + rev_seq_str

    return polindrom_seq_str

def create_polindrome(letters, neg):
    for i in range(500):
        item = create_sequence(letters,neg)
        with open('pos_examples_polindrome', 'a+') as f:
            f.write(''.join([item, '\n']))

def crete_non_polindrome(letters, neg):
    for i in range(500):
        item = create_sequence(letters,neg)
        with open('neg_examples_polindrome', 'a+') as f:
            f.write(''.join([item, '\n']))


def generator_polindromes():
    letters = ['x', 'y', 'z', '1']
    neg = [True, False]

    create_polindrome(letters, neg[0])
    crete_non_polindrome(letters, neg[1])


generate_pattern()
generate_prime()
generator_divider()

generator_polindromes()


