// Quick test to isolate market analytics issue
use std::collections::VecDeque;

fn main() {
    println!("Testing market analytics extraction...");
    
    // Create simple valid candles
    let mut candles = VecDeque::new();
    for i in 0..100 {
        let price = 50000.0 + i as f64 * 10.0;
        candles.push_back((
            price,           // open
            price + 10.0,    // high
            price - 10.0,    // low
            price + 5.0,     // close
            1000.0,          // volume
        ));
    }
    
    println!("Created {} candles", candles.len());
    
    // Test Fourier
    let n = 64;
    let max_k = (n / 2).min(32);
    println!("Testing Fourier with n={}, max_k={}", n, max_k);
    
    let mut count = 0;
    for k in 2..max_k {
        for i in 0..n {
            count += 1;
        }
    }
    println!("Fourier operations: {}", count);
    
    println!("Test complete!");
}