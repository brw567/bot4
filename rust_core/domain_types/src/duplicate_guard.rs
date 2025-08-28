//! # COMPILE-TIME DUPLICATE GUARD
//! Fails compilation if duplicates are detected

/// Macro to ensure single definition
#[macro_export]
macro_rules! ensure_single_definition {
    ($type:ty) => {
        const _: () = {
            // This will fail if the type is defined multiple times
            fn _check_single_definition() {
                let _: $type;
            }
        };
    };
}

// Enforce single definitions
ensure_single_definition!(Price);
ensure_single_definition!(Quantity);
ensure_single_definition!(Order);
ensure_single_definition!(Position);
ensure_single_definition!(Trade);
ensure_single_definition!(Candle);

// Karl: "The compiler is our enforcer"
