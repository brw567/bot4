#!/bin/bash
# Fix all incorrect to_f64_retain() calls to use to_f64().unwrap_or(0.0)
# Team: Sam (Code Quality)

echo "Fixing Decimal conversion methods..."

# Find all files with to_f64_retain and fix them
find /home/hamster/bot4/rust_core -name "*.rs" -type f | while read -r file; do
    if grep -q "to_f64_retain()" "$file"; then
        echo "Fixing: $file"
        
        # First, ensure ToPrimitive is imported if not already
        if ! grep -q "use rust_decimal::prelude::ToPrimitive" "$file"; then
            # Add the import after the first use statement or at the beginning
            sed -i '1a use rust_decimal::prelude::ToPrimitive;' "$file"
        fi
        
        # Replace to_f64_retain() with to_f64().unwrap_or(0.0)
        sed -i 's/\.to_f64_retain()/\.to_f64().unwrap_or(0.0)/g' "$file"
    fi
done

echo "Decimal conversion fix complete!"
