# The MIT License
#
# Copyright (c) OpenAI (https://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import contextlib
import faulthandler
import io
import os
import platform
import signal
import tempfile
import subprocess
from typing import Optional

@contextlib.contextmanager
def swallow_subprocess_output():
    """Context manager to swallow stdout and stderr for subprocesses."""
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def _popen_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            # Avoid setting stdout or stderr if capture_output is True
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_popen(*args, **kwargs)

    def _run_patch(*args, **kwargs):
        if 'capture_output' in kwargs and kwargs['capture_output']:
            # Avoid setting stdout or stderr if capture_output is True
            kwargs.pop('stdout', None)
            kwargs.pop('stderr', None)
        else:
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
        return original_run(*args, **kwargs)

    subprocess.Popen = _popen_patch
    subprocess.run = _run_patch
    try:
        yield
    finally:
        subprocess.Popen = original_popen
        subprocess.run = original_run

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                with swallow_subprocess_output():
                    yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def safe_environment():
    # Save original functions
    original_kill = os.kill
    original_subprocess_call = subprocess.call
    original_check_output = subprocess.check_output

    def safe_kill(pid, sig):
        print(f"Prevented attempt to kill PID {pid} with signal {sig}")
        # Prevent killing any process, or you can add your logic to allow killing non-critical processes

    def safe_subprocess_call(command, *args, **kwargs):
        print(f"Intercepted subprocess call: {command}")
        if 'killall' in command:
            return 0  # Simulate successful execution without doing anything
        return original_subprocess_call(command, *args, **kwargs)

    def safe_check_output(command, *args, **kwargs):
        print(f"Intercepted command: {command}")
        if 'ps' in command:
            return b""  # Simulate no processes found
        return original_check_output(command, *args, **kwargs)

    # Override the risky functions with the safe versions
    os.kill = safe_kill
    subprocess.call = safe_subprocess_call
    subprocess.check_output = safe_check_output

    try:
        yield
    finally:
        # Restore original functions after the block
        os.kill = original_kill
        subprocess.call = original_subprocess_call
        subprocess.check_output = original_check_output


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """
    
    import os
    import time
    from datetime import datetime

    os.environ['TZ'] = 'UTC'
    time.tzset()
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" 
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import matplotlib.pyplot as plt
    plt.close('all')
