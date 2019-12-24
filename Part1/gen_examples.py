
try:
    import rstr
except ImportError:
    import subprocess
    import os
    import sys

    cfolder = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-m", "pip", "install", "--target=" + cfolder, 'rstr']
    subprocess.call(cmd)


good = []
bad = []
for i in range(500):
    good.append(rstr.xeger(r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'))
    bad.append(rstr.xeger(r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+'))

with open('pos_examples', 'w') as f:
    f.write('\n'.join(good))
with open('neg_examples', 'w') as f:
    f.write('\n'.join(bad))
