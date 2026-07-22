with open('tools/permute_qc_report.py', 'r') as f:
    content = f.read()

import re

# We see a huge mess of repeated imports at the top.
# Let's replace the entire head block up to `logging.basicConfig` with a clean one.
# First, let's find `logging.basicConfig`
idx = content.find("logging.basicConfig(level=logging.INFO")

clean_head = '''#!/usr/bin/env python3
import argparse
import base64
import dataclasses
import datetime
import html
import io
import json
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_permute import (  # noqa: E402
    CANONICAL_REGIONS,
    NEAR_GENE_REGIONS,
    MIN_REGION_BULK_N,
    TOLERANCE_MEDIAN_LOG10_RATIO_DIFF,
)

'''

new_content = clean_head + content[idx:]

with open('tools/permute_qc_report.py', 'w') as f:
    f.write(new_content)
