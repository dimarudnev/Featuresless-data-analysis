from functools import wraps
import time
import os
import numpy as np
import math

datadir = 'D:\\Data\\TalkingData'
def log(func):
    @wraps(func)
    def logWrapper(*args, **kwargs):
        print("Start evaluate '{}' function".format(func.__name__))
        start = time.time();
        res = func(*args, **kwargs)
        end = time.time();
        print("End evaluate '{}' function ({} s) ".format(func.__name__, end-start))
        return res;
    return logWrapper

def cache(fromCache=True):
    def cacheDecorator(func):
        @wraps(func)
        def cacheWrapper(*args, **kwargs):
            fileName = os.path.join(datadir, func.__name__ + ".npy")
            if fromCache & os.path.isfile(fileName):
                print("Get {} from cache".format(func.__name__))
                return np.load(fileName);
            else:  
                result = func(*args, **kwargs)
                np.save(fileName, result);
                return result;
        return cacheWrapper
    return cacheDecorator


def jaccardDistance(dict1, dict2):
    a = len(dict1.keys())
    b = len(dict2.keys())
    c = len(dict1.keys() & dict2.keys())
    d= (a+b-c);
    return 1 if d == 0 else 1 - c/d;

def sorensenDistance(dict1, dict2):
    a = len(dict1.keys())
    b = len(dict2.keys())
    c = 2*len(dict1.keys() & dict2.keys())
    d= a + b;
    return 1 if d == 0 else 1 - c/d;
 

def simpsonDistance(dict1, dict2):
    a = len(dict1.keys())
    b = len(dict2.keys())
    c = len(dict1.keys() & dict2.keys())
    d= min(a, b);
    return 1 if d == 0 else 1 - c/d;

def euclideanDictance(dict1, dict2):
    res = 0;
    for key in set(dict1.keys() | dict2.keys()):
        res += (dict1.get(key, 0) - dict2.get(key, 0)) ** 2
    return math.sqrt(res)