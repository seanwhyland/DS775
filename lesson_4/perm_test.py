import pprint as pp
from itertools import permutations
import timeit
import math
import tqdm

routes = list(range(0,20))

# start_time = timeit.default_timer()
p = set(permutations(routes))
print(f"{len(p)}")
# stop_time = timeit.default_timer()

# print('Time elapsed: ', stop_time - start_time)
