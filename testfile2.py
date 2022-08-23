import sys
import os

def get_params(tb):
    while tb.tb_next:
        tb = tb.tb_next
    frame = tb.tb_frame
    code = frame.f_code
    argcount = code.co_argcount
    if code.co_flags & 4: # *args
        argcount += 1
    if code.co_flags & 8: # **kwargs
        argcount += 1
    names = code.co_varnames[:argcount]
    params = {}
    for name in names:
        params[name] = frame.f_locals.get(name, '<deleted>')
    return params


def f(a, b=2, c=3, *d, **e):
    del c
    c = 4
    e['g'] = 6
    assert False

try:
    # f(1, f=5)
    os.system('jupyter-book build /Users/chandrasingh/anaconda3/workingDir/learning/')
except:
    print('>>>> ', get_params(sys.exc_info()[2]))
