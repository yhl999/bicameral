#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ensure upstream is fetched
git fetch upstream main || true

echo "Exporting core patches from local HEAD vs upstream/main..."
mkdir -p patches/graphiti_core

# Read from the allowlist
while IFS= read -r file; do
    # Skip empty lines or comments
    [[ -z "$file" || "$file" == \#* ]] && continue
    
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        patch_file="patches/${file}.patch"
        mkdir -p "$(dirname "$patch_file")"
        
        # generate the patch
        git diff upstream/main HEAD -- "$file" > "$patch_file" || true
        
        # Check if the patch actually has changes
        if [ -s "$patch_file" ]; then
            echo "✅ Exported $patch_file"
        else
            echo "⚠️  No difference for $file (patch file is empty)"
            rm "$patch_file"
        fi
    fi
done < config/graphiti_core_allowlist.txt

echo "Done."
