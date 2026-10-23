"""Entry point so `python -m genenull TABLE` works without a runpy warning."""
from .genenull import _main

if __name__ == '__main__':
    _main()
