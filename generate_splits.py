"""
Generate splits.

This is the script for generating the initial splits of GT-Lejonet_HTR_202602.
It is checked into git for documentation purposes only; we want to reuse the
orignal splits whenever possible.

I generated splits of four sizes, each one including the smaller splits (for
example, the 5% split includes the 2.5% split and an additional 2.5% pages).
The split is done for each subset (any directory containing a `page` directory),
and at least one page is guaranteed to become a test page. This means that small
sets will have a higher weight in the final validation set.
"""

import os
import random

random.seed(10)

splits = [0.025, 0.05, 0.075, 0.1]

for parent, _, files in os.walk("/home/dgxuser/erik/projects/swedish-lion/data/only_ground_truth"):
    if parent.endswith("page"):
        files = [
            file
            for file in files
            if not file.startswith("._") and file.endswith(".xml")
        ]

        # Select a 'guarantee' page that ends up in the test split
        guarantee = random.choice(files)
        sets = [[guarantee] for _ in splits]

        # Then, add files proportinally to the subset's size. 
        for file in files:
            val = random.random()
            for set, split in zip(sets, splits):
                if val < split:
                    set.append(file)

        # Save splits to `test_0025` and so on, in the `page` directory.
        for set, split in zip(sets, ["0025", "0050", "0075", "0100"]):
            path = os.path.join(parent, f"test_{split}")

            if os.path.exists(path):
                print(f"Split {path} exists already! Will not overwrite it.")
                exit()
                

            with open(path, "w") as f:
                f.write("\n".join(set))
