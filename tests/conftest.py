"""Keep test runs away from the real prediction history."""
import os
import tempfile

os.environ["BANANA_HISTORY_DB"] = os.path.join(tempfile.mkdtemp(prefix="banana_test_"), "history.sqlite")
