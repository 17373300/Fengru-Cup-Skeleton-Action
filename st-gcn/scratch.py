import torch
import torch.nn as nn


def func(a, b, **kwargs):
    for kv in kwargs:
        print(kv)


func(a=1, b=2, c=3, d=4)
