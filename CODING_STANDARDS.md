# Bot4 Coding Standards & Architecture Guidelines
## MANDATORY - All Code Must Follow These Standards

---

## 🔴 CRITICAL: These Standards Are NON-NEGOTIABLE

Per Alex's directive: **"All future development MUST follow Software Development Best Practices and must apply good Class and Type Separation"**

Violation of these standards = **IMMEDIATE REJECTION**

---

## 1. Architecture Pattern: Hexagonal (Ports & Adapters)

### Mandatory Structure for ALL New Code

```
rust_core/
├── domain/                 # Core business logic (NO external dependencies)
│   ├── entities/          # Mutable objects with identity
│   │   ├── order.rs       # Order entity
│   │   ├── position.rs    # Position entity
│   │   └── mod.rs
│   ├── value_objects/     # Immutable objects without identity
│   │   ├── price.rs       # Price value object
│   │   ├── quantity.rs    # Quantity value object
│   │   ├── symbol.rs      # Symbol value object
│   │   └── mod.rs
│   ├── services/          # Domain services
│   │   ├── trading_service.rs
│   │   ├── risk_service.rs
│   │   └── mod.rs
│   └── events/            # Domain events
│       ├── order_placed.rs
│       └── position_closed.rs
│
├── application/           # Application services (use cases)
│   ├── commands/         # Command handlers
│   │   ├── place_order.rs
│   │   └── cancel_order.rs
│   ├── queries/          # Query handlers
│   │   ├── get_positions.rs
│   │   └── get_order_history.rs
│   └── services/         # Application services
│       └── trading_orchestrator.rs
│
├── ports/                # Interfaces (traits)
│   ├── inbound/         # Driving ports (use cases)
│   │   ├── trading_use_case.rs
│   │   └── risk_use_case.rs
│   └── outbound/        # Driven ports (infrastructure)
│       ├── exchange_port.rs
│       ├── repository_port.rs
│       └── event_publisher_port.rs
│
├── adapters/            # Interface implementations
│   ├── inbound/        # Driving adapters (controllers)
│   │   ├── rest/
│   │   ├── websocket/
│   │   └── grpc/
│   └── outbound/       # Driven adapters (infrastructure)
│       ├── exchanges/
│       │   ├── binance_adapter.rs
│       │   └── kraken_adapter.rs
│       ├── persistence/
│       │   ├── postgres_repository.rs
│       │   └── redis_cache.rs
│       └── messaging/
│           └── kafka_publisher.rs
│
└── dto/                # Data Transfer Objects
    ├── request/        # API request DTOs
    ├── response/       # API response DTOs
    └── database/       # Database DTOs
```

---

## 2. Class and Type Separation Rules

### A. Domain Models (Core Business)

```rust
// domain/entities/order.rs
// ENTITY: Has identity, mutable
pub struct Order {
    id: OrderId,           // Identity
    symbol: Symbol,        // Value object
    price: Price,          // Value object
    quantity: Quantity,    // Value object
    status: OrderStatus,   // Enum
    created_at: DateTime<Utc>,
}

impl Order {
    // Business logic ONLY
    pub fn can_cancel(&self) -> bool {
        matches!(self.status, OrderStatus::Pending | OrderStatus::Open)
    }
    
    // Domain events
    pub fn place(self) -> Result<(Order, OrderPlaced)> {
        // Business rules validation
        // Return modified order + event
    }
}
```

### B. Value Objects (Immutable)

```rust
// domain/value_objects/price.rs
// VALUE OBJECT: No identity, immutable
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Price(f64);

impl Price {
    pub fn new(value: f64) -> Result<Self> {
        if value <= 0.0 {
            return Err(DomainError::InvalidPrice);
        }
        Ok(Price(value))
    }
    
    pub fn value(&self) -> f64 {
        self.0
    }
    
    // Business operations
    pub fn apply_slippage(&self, slippage: f64) -> Price {
        Price(self.0 * (1.0 + slippage))
    }
}
```

### C. DTOs (Data Transfer)

```rust
// dto/request/place_order_dto.rs
// DTO: For external communication ONLY
#[derive(Deserialize, Validate)]
pub struct PlaceOrderDto {
    #[validate(length(min = 1))]
    pub symbol: String,
    
    #[validate(range(min = 0.0))]
    pub price: f64,
    
    #[validate(range(min = 0.0))]
    pub quantity: f64,
}

// Mapping to domain
impl TryFrom<PlaceOrderDto> for Order {
    type Error = ValidationError;
    
    fn try_from(dto: PlaceOrderDto) -> Result<Self, Self::Error> {
        Ok(Order::new(
            Symbol::new(&dto.symbol)?,
            Price::new(dto.price)?,
            Quantity::new(dto.quantity)?,
        ))
    }
}
```

### D. Ports (Interfaces)

```rust
// ports/outbound/exchange_port.rs
// PORT: Interface only, no implementation
#[async_trait]
pub trait ExchangePort: Send + Sync {
    async fn place_order(&self, order: &Order) -> Result<OrderId>;
    async fn cancel_order(&self, id: &OrderId) -> Result<()>;
    async fn get_order_status(&self, id: &OrderId) -> Result<OrderStatus>;
}

// ports/outbound/repository_port.rs
#[async_trait]
pub trait OrderRepository: Send + Sync {
    async fn save(&self, order: &Order) -> Result<()>;
    async fn find_by_id(&self, id: &OrderId) -> Result<Option<Order>>;
    async fn find_active(&self) -> Result<Vec<Order>>;
}
```

### E. Adapters (Implementations)

```rust
// adapters/outbound/exchanges/binance_adapter.rs
// ADAPTER: Implements port interface
pub struct BinanceAdapter {
    client: BinanceClient,
    rate_limiter: RateLimiter,
}

#[async_trait]
impl ExchangePort for BinanceAdapter {
    async fn place_order(&self, order: &Order) -> Result<OrderId> {
        // Convert domain to exchange API format
        let api_request = self.to_api_format(order);
        
        // Rate limiting
        self.rate_limiter.acquire().await?;
        
        // API call
        let response = self.client.place_order(api_request).await?;
        
        // Convert response to domain
        Ok(OrderId::from(response.order_id))
    }
}
```

---

## 3. SOLID Principles (MANDATORY)

### S - Single Responsibility
✅ **CORRECT:**
```rust
// Each class has ONE reason to change
pub struct OrderValidator {
    // ONLY validates orders
}

pub struct OrderExecutor {
    // ONLY executes orders
}
```

❌ **WRONG:**
```rust
pub struct OrderManager {
    // Does validation AND execution AND persistence
}
```

### O - Open/Closed
✅ **CORRECT:**
```rust
// Open for extension via traits
pub trait TradingStrategy {
    fn evaluate(&self, market: &MarketData) -> Signal;
}

// Closed for modification - add new strategies without changing existing
pub struct MomentumStrategy;
impl TradingStrategy for MomentumStrategy { }

pub struct MeanReversionStrategy;
impl TradingStrategy for MeanReversionStrategy { }
```

### L - Liskov Substitution
✅ **CORRECT:**
```rust
// Subtypes must be substitutable
trait Exchange {
    async fn place_order(&self, order: Order) -> Result<OrderId>;
}

// All implementations honor the contract
impl Exchange for Binance { }
impl Exchange for Kraken { }
```

### I - Interface Segregation
✅ **CORRECT:**
```rust
// Small, focused interfaces
trait OrderReader {
    async fn find_by_id(&self, id: OrderId) -> Result<Option<Order>>;
}

trait OrderWriter {
    async fn save(&self, order: Order) -> Result<()>;
}
```

❌ **WRONG:**
```rust
trait OrderRepository {
    // Too many responsibilities in one interface
    async fn save();
    async fn update();
    async fn delete();
    async fn find();
    async fn find_all();
    async fn count();
    // etc...
}
```

### D - Dependency Inversion
✅ **CORRECT:**
```rust
// High-level module depends on abstraction
pub struct TradingService {
    exchange: Box<dyn ExchangePort>,  // Depends on interface
    repository: Box<dyn OrderRepository>,
}

// Low-level module implements abstraction
pub struct BinanceAdapter;
impl ExchangePort for BinanceAdapter { }
```

---

## 4. Design Patterns (REQUIRED)

### Repository Pattern (ALL data access)
```rust
#[async_trait]
pub trait Repository<T, ID> {
    async fn save(&self, entity: &T) -> Result<()>;
    async fn find_by_id(&self, id: &ID) -> Result<Option<T>>;
    async fn find_all(&self) -> Result<Vec<T>>;
    async fn delete(&self, id: &ID) -> Result<()>;
}

pub struct PostgresOrderRepository {
    pool: PgPool,
}

#[async_trait]
impl Repository<Order, OrderId> for PostgresOrderRepository {
    // Implementation
}
```

### Command Pattern (ALL operations)
```rust
#[async_trait]
pub trait Command {
    type Output;
    async fn execute(&self) -> Result<Self::Output>;
}

pub struct PlaceOrderCommand {
    order: Order,
    exchange: Arc<dyn ExchangePort>,
    repository: Arc<dyn OrderRepository>,
}

#[async_trait]
impl Command for PlaceOrderCommand {
    type Output = OrderId;
    
    async fn execute(&self) -> Result<Self::Output> {
        // Validate
        self.order.validate()?;
        
        // Execute on exchange
        let id = self.exchange.place_order(&self.order).await?;
        
        // Persist
        self.repository.save(&self.order).await?;
        
        Ok(id)
    }
}
```

### Factory Pattern (Object creation)
```rust
pub trait OrderFactory {
    fn create_market_order(&self, symbol: Symbol, quantity: Quantity) -> Order;
    fn create_limit_order(&self, symbol: Symbol, price: Price, quantity: Quantity) -> Order;
}
```

---

## 5. Testing Standards

### Test Structure
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    // Unit tests for business logic
    mod unit_tests {
        #[test]
        fn should_validate_positive_price() { }
        
        #[test]
        fn should_reject_negative_price() { }
    }
    
    // Integration tests for adapters
    mod integration_tests {
        #[tokio::test]
        async fn should_place_order_on_exchange() { }
    }
    
    // Property-based tests
    mod property_tests {
        use quickcheck::{quickcheck, TestResult};
        
        #[quickcheck]
        fn price_should_always_be_positive(value: f64) -> TestResult { }
    }
}
```

### Test Naming Convention
```rust
// Pattern: should_[expected_behavior]_when_[condition]
#[test]
fn should_return_error_when_price_is_negative() { }

#[test]
fn should_cancel_order_when_status_is_pending() { }
```

---

## 6. Error Handling

### Domain Errors
```rust
#[derive(thiserror::Error, Debug)]
pub enum DomainError {
    #[error("Invalid price: {0}")]
    InvalidPrice(f64),
    
    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance {
        required: f64,
        available: f64,
    },
}
```

### Application Errors
```rust
#[derive(thiserror::Error, Debug)]
pub enum ApplicationError {
    #[error("Domain error: {0}")]
    Domain(#[from] DomainError),
    
    #[error("Infrastructure error: {0}")]
    Infrastructure(#[from] InfrastructureError),
}
```

---

## 7. Documentation Requirements

### Every Public Item MUST Have:
```rust
/// Brief description of what this does
/// 
/// # Arguments
/// * `order` - The order to place
/// 
/// # Returns
/// * `Ok(OrderId)` - The ID of the placed order
/// * `Err(TradingError)` - If placement fails
/// 
/// # Example
/// ```
/// let order = Order::new(symbol, price, quantity);
/// let id = exchange.place_order(order).await?;
/// ```
pub async fn place_order(&self, order: Order) -> Result<OrderId>
```

---

## 8. Performance Standards

### Hot Path Requirements
```rust
// Mark hot paths explicitly
#[inline(always)]
pub fn calculate_pnl(&self, current_price: f64) -> f64 {
    // Zero allocations allowed
    // No heap allocations
    // Prefer stack allocation
}

// Use const generics for compile-time optimization
pub struct OrderBook<const DEPTH: usize> {
    bids: [Order; DEPTH],
    asks: [Order; DEPTH],
}
```

---

## 9. Concurrency Standards

### Use Arc for Shared State
```rust
pub struct TradingEngine {
    orders: Arc<RwLock<HashMap<OrderId, Order>>>,
    exchange: Arc<dyn ExchangePort>,
}
```

### Prefer Channels for Communication
```rust
use tokio::sync::mpsc;

pub struct OrderProcessor {
    receiver: mpsc::Receiver<Order>,
    sender: mpsc::Sender<ExecutionReport>,
}
```

---

## 10. Code Review Checklist

Before ANY PR:
- [ ] Follows hexagonal architecture
- [ ] Proper DTO/Domain separation
- [ ] SOLID principles applied
- [ ] Design patterns used correctly
- [ ] 100% test coverage
- [ ] No fake implementations
- [ ] Documentation complete
- [ ] Performance requirements met
- [ ] Error handling comprehensive
- [ ] No hardcoded values

---

## ENFORCEMENT

**These standards are MANDATORY. No exceptions.**

Every PR will be reviewed against these standards. Non-compliance = REJECTION.

Team Leaders:
- **Alex**: Overall compliance
- **Sam**: Code quality & patterns
- **Morgan**: Domain modeling
- **Jordan**: Performance standards

---

*Last Updated: 2024-01-XX*
*Approved by: Alex (Team Lead)*
*Status: ENFORCED*