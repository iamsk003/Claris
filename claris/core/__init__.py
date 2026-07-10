"""CLARIS core engine.

Pure, sync-safe, no I/O side effects beyond the cache dir. Depends on nothing in
``claris.agent`` or ``claris.api``. The engine never knows which surface called it.
"""
