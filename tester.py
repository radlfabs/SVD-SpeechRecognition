# -*- coding: utf-8 -*-
"""
@author: P.Schwarz
"""

from number_classifier import *
from util import tracking_start, tracing_mem
import cProfile
import pstats
from pstats import SortKey

# Profiling documentation: https://docs.python.org/3/library/profile.html

if __name__ == "__main__":
    os.makedirs(os.path.join(os.getcwd(), "profiler"), exist_ok=True)
    tracking_start()
    cProfile.run('digit_classifier(size=10, k_list=[1500])', 'profiler/tester_stats')
    tracing_mem()

    p = pstats.Stats('profiler/tester_stats')
    p.strip_dirs()
    p.sort_stats(SortKey.TIME, SortKey.CUMULATIVE)
    p.print_stats(20)

