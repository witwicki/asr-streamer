import os, sys, io, contextlib

def suppress_terminal_output(func):
    """ A useful decorator to be applied to instance methods that take self.verbose=False as a cue to suppress all output """
    def wrapper(self, *args, **kwargs):
        if not self.verbose:
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    # additional supression of stderrs
                    # For more details: https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    old_stderr = os.dup(2)
                    sys.stderr.flush()
                    os.dup2(devnull, 2)
                    os.close(devnull)
                    wrapped_func = func(self, *args, **kwargs)
                    os.dup2(old_stderr, 2)
                    os.close(old_stderr)
                    return wrapped_func
        return func(self, *args, **kwargs)
    return wrapper
