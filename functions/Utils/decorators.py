# ==================================================
#  Useful decorators
#
# By Gustavo Patow
# ==================================================
import scipy.io as sio
from pathlib import Path
import functools
import time


# ==================================================
# timer decorator:
# not used... yet! ;-)
def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        # print("   Start timing...")
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"   Finished Timing {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


# ==================================================
# loadOrCompute decorator:
# useful for not reating lengthy numpy calculations. It "eats" the last parameter of the arguments passed on,
# which should be a string. Using this decorator forces that the return value of the decorated function is a
# dictionary of {name: value} pairts, so when the computation was done before and the file exists and is
# loaded we could directly return the Matlab file contents directly. It uses the .mat file format for all
# operations.
def loadOrCompute(func):
    @functools.wraps(func)
    def loading_decorator(*args, **kwargs):
        if not Path(args[-1]).is_file():
            print(f"Computing (@loadOrCompute): {args[-1]}")
            value = tuple(a for a in list(args)[:-1])
            result = func(*value)
            sio.savemat(args[-1], result)
        else:
            print(f"Loading file (@loadOrCompute): {args[-1]} !!!")
            result = sio.loadmat(args[-1])
        return result
    return loading_decorator


# ==================================================
cache = {}

def loadCache(filePath):
    global cache

    if Path(filePath).is_file():
        print('loading cache:', filePath)
        loaded = sio.loadmat(filePath)
        for key, value in loaded.items():
            if key not in ['__header__', '__version__', '__globals__']:
                cache[eval(key)] = value.flatten()
    else:
        print("There is no cache to load")


evalCounter = 0
def vectorCache(cachePath):
    def inner_function(func):
        @functools.wraps(func)
        def vectorCache_wrapper(*args):
            # if evalCounter % 100 == 0: print(" -> TIME: {} seconds ---".format(time.clock() - start_time), flush=True)
            global cache
            key = tuple(args[0].tolist())
            if key in cache:
                # if np.array_equal(x,vect):
                print('.', end='', flush=True)
                return cache[key]
            global evalCounter; evalCounter += 1; print(f"{evalCounter}" if evalCounter % 10 == 0 else "!", end='', flush=True);
            result = func(*args)
            cache[key] = result
            sio.savemat(cachePath, cache)
            return result
        return vectorCache_wrapper
    return inner_function


# ==================================================
# ==================================================
if __name__ == '__main__':
    import numpy as np

    @loadOrCompute
    def test(aParm, aSecond):
        print(f"Computing test on {aParm} and {aSecond}")
        return {'data': np.array([1, 2, 3])}

    kk = test('first', 'second', '../../Data_Produced/AD/test.mat')
    print(f"Result {kk['data'].flatten()}")
# ==================================================
# ==================================================
# ==================================================EOF
