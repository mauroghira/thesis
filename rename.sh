#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PARENT_DIR=~/thesis/Spiral_pattern

# Dry-run by default. Pass --go to actually rename.
DRY_RUN=false
if [[ "${1-}" == "--go" ]]; then DRY_RUN=false; fi

for subdir in "$PARENT_DIR"/*/; do
  [[ -d "$subdir" ]] || continue
  name=$(basename "$subdir")
  echo "Entering $subdir"

  files=( "$subdir"/m"${name}"beta10_000* )
  ((${#files[@]})) || { echo "  (no matches)"; continue; }

  for file in "${files[@]}"; do
    base=$(basename "$file")

    # Match: m<dir>_000([digits])[anything][.ext]
    if [[ "$base" =~ ^m${name}beta10_000([0-9]{2})[^\.]*(\..*)?$ ]]; then
      num="${BASH_REMATCH[1]}"
      ext="${BASH_REMATCH[2]:-}"   # extension if present

      tens=${num:0:1}
      ones=${num:1:1}
 
      if (( 10#$tens > 1 )); then
        tens=$((tens - 2))
      fi

      newnum=$(( 10#$tens * 10 + 10#$ones ))

      # drop leading zero if you don’t want padding:
      newnum=$((10#$newnum))
      newbase="img_${name}_${newnum}${ext}"
      newpath="$subdir/$newbase"

      if [[ -e "$newpath" ]]; then
        echo "  WARNING: target exists, skipping: $newbase"
        continue
      fi

      echo "  Renaming: $base -> $newbase"
      if [[ "$DRY_RUN" != "true" ]]; then
        mv -- "$file" "$newpath"
      fi
    else
      echo "  Skipping (pattern mismatch): $base"
    fi
  done
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo
  echo "Dry-run complete. Run again with '--go' (or set DRY_RUN=false) to apply changes."
fi
