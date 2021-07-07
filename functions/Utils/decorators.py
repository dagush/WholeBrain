# ==================================================
#  Useful decorators
#
# By Gustavo Patow
# ==================================================
import numpy as np
import scipy.io as sio
from pathlib import Path
import functools
import time


verbose = True


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
# Simple methods to delete entries that are not useful, which happens when reading the data from a
# file instead of computing it...
def cleanDict(resData):
    result = resData.pop('__header__', None)
    result = resData.pop('__version__', None)
    result = resData.pop('__globals__', None)
    return resData

# ==================================================
# loadOrCompute decorator:
# useful for not creating lengthy numpy calculations. It "eats" the last parameter of the arguments passed on,
# which should be a string. Using this decorator forces that the return value of the decorated function is a
# dictionary of {name: value} pairs, so when the computation was done before and the file exists and is
# loaded we could directly return the Matlab file contents directly. It uses the .mat file format for all
# operations.
def loadOrCompute(func):
    @functools.wraps(func)
    def loading_decorator(*args, **kwargs):
        if not Path(args[-1]).is_file():
            if verbose: print(f"Computing (@loadOrCompute): {args[-1]}", flush=True)
            value = tuple(a for a in list(args)[:-1])
            result = func(*value)
            sio.savemat(args[-1], result)
        else:
            if verbose: print(f"Loading file (@loadOrCompute): {args[-1]} !!!", flush=True)
            result = cleanDict(sio.loadmat(args[-1], squeeze_me=True))
        return result
    return loading_decorator


# ==================================================
# vectorCache decorator:
# Stores a vector of values in a given file, updating its contents each time it is called.
# In general, it works much like the loadOrCompute decorator...
# There are two possible initializations: one using loadCache, to directly load a single vector file,
# or using loadMultipleCache, to load all files within a range and merging their contents into a single cache.
def loadMultipleCache(filePath, valueRange, useDecimals=3, fileNameDecimals=3):
    for we in valueRange:
        fileName = filePath.format(np.round(we, decimals=fileNameDecimals)+0.0)  # add 0.0 to prevent stupid negative -0.0 result from round...
        loadSingleCache(fileName, useDecimals=useDecimals)


cachePath = None
cache = {}
decimals = 3
def loadSingleCache(filePath, useDecimals=3):
    global cache, decimals, cachePath
    cachePath = filePath
    decimals = useDecimals
    if Path(filePath).is_file():
        print('Cache decorator: loading cache:', filePath)
        loaded = sio.loadmat(filePath, squeeze_me=True)
        for key, value in loaded.items():
            if key not in ['__header__', '__version__', '__globals__']:
                realKey = tuple(np.round(np.asarray(eval(key)), decimals).tolist())
                cache[realKey] = value.flatten()
    else:
        print(f"Cache decorator: There is no cache to load: {filePath}")


evalCounter = 0
def vectorCache(filePath):
    def inner_function(func):
        @functools.wraps(func)
        def vectorCache_wrapper(*args):
            if verbose:
                global evalCounter; evalCounter += 1; print(f"{evalCounter} -> ", end='', flush=True)
            global cache
            key = tuple(np.round(args[0],decimals).tolist())
            if key in cache:
                if verbose: print(f' {cache[key]} (loaded)', flush=True)
                return cache[key]
            result = func(*args)
            cache[key] = result
            if verbose: print(f' {result} (Computed)', flush=True)
            if filePath is None:
                sio.savemat(cachePath, cache)  # Use the original one we started with
            else:
                sio.savemat(filePath, cache)  # Use this one...
            return result
        return vectorCache_wrapper
    return inner_function


# ==================================================
# ==================================================
if __name__ == '__main__':
    # ====================== some debug code...
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
