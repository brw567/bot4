#!/bin/bash
# Fix incorrectly replaced exchange_name references
# Team: Sam (Code Quality)

echo "Fixing exchange_name references..."

file="/home/hamster/bot4/rust_core/crates/order_management/src/router.rs"

# First backup the file
cp "$file" "$file.bak"

# Fix the struct field
sed -i 's/pub &exchange_name:/pub exchange_name:/' "$file"

# Fix function parameters
sed -i 's/pub fn new(&exchange_name:/pub fn new(exchange_name:/' "$file"
sed -i 's/pub fn remove_route(&self, &exchange_name:/pub fn remove_route(\&self, exchange_name:/' "$file"
sed -i 's/pub fn set_primary_exchange(&self, &exchange_name:/pub fn set_primary_exchange(\&self, exchange_name:/' "$file"

# Fix in function bodies - replace self.&exchange_name with self.exchange_name
sed -i 's/self\.&exchange_name/self.exchange_name/g' "$file"

# Fix route.&exchange_name to route.exchange_name
sed -i 's/route\.&exchange_name/route.exchange_name/g' "$file"

# Fix standalone &exchange_name to exchange_name (but keep & for references)
sed -i 's/for (&exchange_name,/for (exchange_name,/' "$file"
sed -i 's/        &exchange_name:/        exchange_name:/' "$file"
sed -i 's/        &exchange_name,/        exchange_name,/' "$file"
sed -i 's/(&exchange_name,/(exchange_name,/' "$file"
sed -i 's/remove(&exchange_name)/remove(exchange_name)/' "$file"
sed -i 's/= Some(&exchange_name)/= Some(exchange_name)/' "$file"
sed -i 's/get_mut(&exchange_name)/get_mut(exchange_name)/' "$file"

echo "Fixed exchange_name references!"
