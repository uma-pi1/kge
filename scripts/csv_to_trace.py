#!/usr/bin/env python
import sys
import pandas as pd
import yaml

df = pd.read_csv(sys.stdin)
df.apply(lambda row: print(yaml.dump(row.to_dict(), width=float("inf"), default_flow_style=True).strip()), axis=1)
