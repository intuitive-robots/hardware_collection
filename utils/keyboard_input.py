import sys
import select
import tty
import termios

class NonBlockingKeyPress(object):
    """
    This class was copied and adapted from: https://stackoverflow.com/a/10079805
    Note that this solution is sometimes confused when spamming a character and that there are problems with special characters such as arrow keys.
    """

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            # Read one character
            data = sys.stdin.read(1)
            # Flush received but not read and written but not transmitted data
            # This does no fully fix that the input is confused if a key is spammed
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
            return data
        return False