#!/bin/bash

# Parent directory containing subdirectories
PARENT_DIR=~/thesis/Spiral_pattern

# Iterate over all directories inside PARENT_DIR
for dir in "$PARENT_DIR"/*/; do
    if [ -d "$dir" ]; then
        echo "Entering $dir"
        name=$(basename "$dir")

        # Build arguments that don't depend on y#
        para="$PARENT_DIR/ref4.1_3D.para"

        # Iterate over years 1–10
        for year in $(seq 1 10); do
            ydir="$dir/inc_0/y$year"
            
            # Ensure directory exists
            mkdir -p "$ydir"
            cd "$ydir" || exit 1

            echo "  Running in $ydir"

            phantom_file="$dir/img_${name}_${year}"

            # Run mcfost
            mcfost "$para" -phantom "$phantom_file" -scale_length_units 5
            mcfost "$para" -phantom "$phantom_file" -scale_length_units 5 -img 1300

            cd - > /dev/null   # return to previous directory quietly
        done
    fi
done
