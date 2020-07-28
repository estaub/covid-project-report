import pathlib
from functools import reduce

def mostRecentTsv():
    def reducer(accum, v):
        if accum is None or accum.name < v.name:
            accum = v
        return accum
    return reduce(reducer,list(pathlib.Path().glob('./????-??-??.tsv')), None)

print(mostRecentTsv())
