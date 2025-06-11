#!/bin/bash

# Usage: ./shift_z.sh filename.gro
# Output: filename_z.gro

if [[ $# -ne 1 || "$1" != *.gro ]]; then
  echo "Usage: $0 filename.gro"
  exit 1
fi

input="$1"
output="${input%.gro}_z.gro"

awk '
NR <= 2 { print; next }   # Print header and atom count unchanged

{
  lines[NR] = $0         # Save all lines to array for later
}

END {
  # Process all but last line (atoms)
  for (i = 3; i < NR; i++) {
    line = lines[i]

    # Extract fixed-width fields from line
    rest = substr(line, 1, 20)
    x = substr(line, 21, 8) + 0
    y = substr(line, 29, 8) + 0
    z = substr(line, 37, 8) + 0 + 2  # Add 2 to Z

    # Print with formatting (same widths as original)
    printf "%s%8.3f%8.3f%8.3f\n", rest, x, y, z
  }

  # Process last line: only add 4 to the last number, keep rest of line intact

  last_line = lines[NR]

  # Extract last number from the last line using match and substr
  # Assuming last number is the last float in the line:
  # Find last number start position:
  match(last_line, /[0-9.+-]+$/)
  last_num_start = RSTART
  last_num = substr(last_line, last_num_start)

  # Convert to number and add 4
  new_last_num = last_num + 4

  # Print the line up to before last number, then new number
  printf "%s%.5f\n", substr(last_line, 1, last_num_start - 1), new_last_num
}
' "$input" > "$output"

echo "Wrote shifted file to: $output"

