#!/usr/bin/env bash
set -euo pipefail

echo "Testing tens-digit rule (if tens>1 subtract 2):"
for num in 00 01 02 08 09 10 15 20 25 32 99; do
  tens=${num:0:1}
  ones=${num:1:1}

  # force base-10 conversion to avoid octal problems
  tens_i=$((10#$tens))
  ones_i=$((10#$ones))

  if (( tens_i > 1 )); then
    tens_i=$((tens_i - 2))
  fi

  new_i=$((tens_i * 10 + ones_i))

  # print both integer and zero-padded forms
  printf "orig=%s -> new_int=%d -> new_padded=%02d\n" "$num" "$new_i" "$new_i"
done
