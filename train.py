# Read in file
f = open('result.txt')
lines = f.readlines()
f.close()

d = {}
lastResult = lines[-2]
import re
lastResult = re.sub(r'[\W+.]', ' ', lastResult)
print lastResult.split()
(key, val) = lastResult.split()
d[key] = val

print d