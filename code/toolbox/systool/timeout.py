#! /usr/bin/python
# -*- coding:utf-8 -*-
__author__ = 'hkh'
__date__ = '25/10/2016'
__version__ = 1.0

import signal
import functools
from .system_tool import KThread

class TimeoutError(Exception):
    """function run timeout"""


def timeoutThread(seconds, msg=None):
    """超时装饰器，指定超时时间
    若被装饰的方法在指定的时间内未返回，则抛出Timeout异常
    """
    def timeout_decorator(func):
        """真正的装饰器"""

        def _new_func(oldfunc, result, oldfunc_args, oldfunc_kwargs):
            result.append(oldfunc(*oldfunc_args, **oldfunc_kwargs))

        def _(*args, **kwargs):
            result = []
            new_kwargs = { # create new args for _new_func, because we want to get the func return val to result list
                'oldfunc': func,
                'result': result,
                'oldfunc_args': args,
                'oldfunc_kwargs': kwargs
            }
            thd = KThread(target=_new_func, args=(), kwargs=new_kwargs)
            thd.start()
            thd.join(seconds)
            alive = thd.isAlive()
            thd.kill() # kill the child thread
            if alive:
                if msg is None:
                    raise TimeoutError(u'function run too long, timeout %d seconds.' % seconds)
                else:
                    raise TimeoutError(msg)
            else:
                return result[0]
        _.__name__ = func.__name__
        _.__doc__ = func.__doc__
        return _
    return timeout_decorator

def timeoutSignal(seconds, error_message = 'Function call timed out'):
    def decorated(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return functools.wraps(func)(wrapper)

    return decorated