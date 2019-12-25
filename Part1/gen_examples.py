
try:
    #import rstr
    from xeger import Xeger
except ImportError:
    import subprocess
    import os
    import sys

    cfolder = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-m", "pip", "install", "--target=" + cfolder, 'xeger']
    subprocess.call(cmd)
    from xeger import Xeger


good = []
bad = []
x = Xeger(limit=20)
for i in range(500):
    good.append(x.xeger(r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'))
    bad.append(x.xeger(r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+'))

with open('pos_examples_new', 'w') as f:
    f.write('\n'.join(good))
with open('neg_examples_new', 'w') as f:
    f.write('\n'.join(bad))
