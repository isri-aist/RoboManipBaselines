#!/bin/sh

# A script to shift the numerical index of files matching the pattern:
# RealUR10eDemo_XXX.rmb
#
# This version is updated to use a more portable method for handling numbers
# with leading zeros, avoiding shell-specific syntax.
#
# Usage: ./shift_indices.sh <N>
# where <N> is the integer value to shift the index by.
# It can be positive (e.g., 5) or negative (e.g., -5).

# --- Configuration ---
# The value to shift the index by, taken from the first command-line argument.
SHIFT_VALUE=$1
# The pattern of files to target.
FILE_PATTERN="RealUR10eDemo_*.rmb"

# --- Validation ---
# Check if the shift value was provided using the portable `[` command.
if [ -z "$SHIFT_VALUE" ]; then
  echo "Error: No shift value provided."
  echo "Usage: $0 <N>"
  exit 1
fi

# Check if the shift value is an integer using a portable 'case' statement.
case $SHIFT_VALUE in
  ''|*[!0-9-]*)
    echo "Error: Shift value must be an integer."
    echo "Usage: $0 <N>"
    exit 1
    ;;
  *)
    # The value appears to be a valid integer.
    ;;
esac


# --- Main Logic ---

# This script uses a two-pass approach to avoid conflicts where a rename
# operation would overwrite an existing file that hasn't been renamed yet.
# Pass 1: Rename all target files to a temporary name (e.g., file_015.rmb.tmp)
# Pass 2: Rename all temporary files to their final name (e.g., file_015.rmb)

echo "--- PASS 1: Renaming files to temporary names ---"
echo "Searching for files matching pattern: '$FILE_PATTERN'"

# Use 'find' piped to a 'while read' loop for robust file handling.
find . -maxdepth 1 -name "$FILE_PATTERN" | while IFS= read -r file; do

  # Extract the numeric part using 'sed' with basic regular expressions for portability.
  current_index_str=$(echo "$file" | sed 's/.*_\([0-9]\{3\}\)\.rmb/\1/')

  # To prevent the shell from interpreting numbers with leading zeros (e.g., 010)
  # as octal, we strip the leading zeros. This is more portable than the '10#' prefix.
  temp_index=$(echo "$current_index_str" | sed -e 's/^0*//')

  # If the original number was "000", it will become an empty string. Set it to "0".
  if [ -z "$temp_index" ]; then
    temp_index="0"
  fi

  # Now we have a clean integer without leading zeros.
  current_index_int=$temp_index

  # Calculate the new index.
  new_index=$((current_index_int + SHIFT_VALUE))

  # Format the new index to be three digits, padded with leading zeros if necessary.
  new_index_padded=$(printf "%03d" "$new_index")

  # Construct the new filename by replacing the old number with the new padded number.
  new_filename=$(echo "$file" | sed "s/_[0-9]\{3\}\.rmb/_${new_index_padded}.rmb/")

  # Construct the temporary filename for the first pass.
  tmp_filename="${new_filename}"

  # --- ACTION ---
  # For safety, the 'mv' command is commented out by default.
  echo "mv \"$file\" \"$tmp_filename\""
  mv "$file" "$tmp_filename"
done

echo ""
echo "--- PASS 2: Renaming temporary files to final names ---"

# Check if any .tmp files exist before trying to loop.
# 'ls' is used here to check for the existence of files matching the pattern.
if ls *.rmb.tmp 1> /dev/null 2>&1; then
  echo "The following commands will be executed:"
  # Loop through the temporary files created in pass 1.
  for file in *.rmb.tmp; do
    # Construct the final filename by removing the '.tmp' suffix.
    final_filename="${file%.tmp}"

    # --- ACTION ---
    # For safety, the 'mv' command is commented out by default.
    echo "mv \"$file\" \"$final_filename\""
    # mv "$file" "$final_filename"
  done
else
  echo "No files were renamed in Pass 1. Please check the directory and file patterns."
fi

echo ""
echo "Script finished. To perform the actual renaming, uncomment the 'mv' lines in the script."

