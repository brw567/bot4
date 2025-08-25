use anyhow::Result;

pub struct SchemaRegistry {
    // Implementation coming next
}

impl SchemaRegistry {
    pub async fn new(_brokers: &str) -> Result<Self> {
        Ok(Self {})
    }
}