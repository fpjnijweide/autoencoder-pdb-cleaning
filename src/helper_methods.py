import logging
import os
import signal


def find(inputlist, search, key=lambda z: z):
    for i in range(len(inputlist)):
        if key(inputlist[i]) == search:
            return i
    return None


def str_noneguard(obj):
    if hasattr(obj, '__name__'):
        return obj.__name__
    if obj is None:
        return ''
    if isinstance(obj, list):
        return str([str_noneguard(x) for x in obj])
    return str(obj)


def freeze(d):
    # thanks to https://stackoverflow.com/a/13264725
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, list):
        return tuple(freeze(value) for value in d)
    return d


class DelayedKeyboardInterrupt:
    # Original author: Gary van der Merwe at https://stackoverflow.com/a/21919644
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def clean_directory():
    dir_name = os.getcwd()
    test = os.listdir(os.path.join(os.path.dirname(__file__), "..", "input_data"))

    for item in test:
        if item.endswith(".pdb") or item.endswith(".df") or item.endswith(".pkl"):
            os.remove(os.path.join(dir_name, item))


def clean(*args):
    clean_directory()
    os._exit(0)
