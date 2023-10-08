import numpy as np

def my_round(val_in):
    integer = np.floor(np.abs(val_in))
    decimal = np.abs(val_in)-integer
    if decimal>=0.5:
        integer += 1
    if val_in<0:
        integer = -integer
    return int(integer)