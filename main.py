"""AlchemyDetect entry point — delegates to alchemydetect.app.main.

(``main()`` calls ``multiprocessing.freeze_support()`` itself, so it is covered
for both this entry point and the ``alchemydetect`` gui-script.)
"""

from alchemydetect.app import main

if __name__ == "__main__":
    main()
