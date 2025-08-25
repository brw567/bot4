use anyhow::Result;
use crate::producers::MarketEvent;

pub struct ClickHouseSink {
    // Implementation coming next
}

impl ClickHouseSink {
    pub async fn new() -> Result<Self> {
        Ok(Self {})
    }
    
    pub async fn write(&self, _event: MarketEvent) -> Result<()> {
        // Implementation coming next
        Ok(())
    }
    
    pub async fn flush(&self) -> Result<()> {
        Ok(())
    }
}