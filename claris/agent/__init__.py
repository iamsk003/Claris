"""Batch captioning agent.

Reads /input/tasks.json, writes /output/results.json, exits cleanly with code 0.
Never crashes on a bad task — a bad clip yields a degraded caption set, not a failure.
"""
