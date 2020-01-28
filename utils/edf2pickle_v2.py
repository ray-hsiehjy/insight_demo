from utils_v2 import Edf_to_PickledArray
from concurrent.futures import ProcessPoolExecutor, as_completed

args = [i for i in range(1, 25, 1)]

with ProcessPoolExecutor() as executor:

    results = [executor.submit(Edf_to_PickledArray, arg) for arg in args]

    for future in as_completed(results):
        print(future.result())
