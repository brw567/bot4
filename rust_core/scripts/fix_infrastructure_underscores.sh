#!/bin/bash

# Fix all underscore issues in infrastructure crate
# Team: Sam (Code Quality)

set -e

echo "Fixing infrastructure crate underscores..."

# Fix all underscore prefixes in infrastructure - they're all actually used
find crates/infrastructure -name "*.rs" -type f | while read file; do
    # Remove underscore prefix from all let bindings that are actually used
    sed -i 's/let _\([a-z_][a-z0-9_]*\) =/let \1 =/g' "$file"
    
    # Also fix function parameters that were incorrectly prefixed
    sed -i 's/fn [a-z_]*(\s*_\([a-z_][a-z0-9_]*\):/fn \1(/g' "$file"
done

echo "Done fixing infrastructure crate"