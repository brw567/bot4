// Minimal test to find hanging issue
fn main() {
    println!("Starting minimal test...");
    
    // Test 1: VecDeque creation
    let mut candles = std::collections::VecDeque::new();
    println!("Created VecDeque");
    
    // Test 2: Simple loop
    for i in 0..100 {
        candles.push_back(i);
    }
    println!("Added 100 items");
    
    // Test 3: Test iteration
    let count = candles.iter().count();
    println!("Counted {} items", count);
    
    // Test 4: Test take
    let taken: Vec<_> = candles.iter().take(50).collect();
    println!("Took {} items", taken.len());
    
    println!("Minimal test complete!");
}