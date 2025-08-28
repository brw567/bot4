//! Duplication detection module

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct DuplicationDetector {
    registry: Arc<RwLock<HashMap<String, HashSet<String>>>>,
}

impl DuplicationDetector {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn update_registry(&self, component: &str, locations: &[String]) -> Result<()> {
        let mut registry = self.registry.write().await;
        let entry = registry.entry(component.to_string()).or_insert_with(HashSet::new);
        
        for location in locations {
            entry.insert(location.clone());
        }
        
        Ok(())
    }
    
    pub async fn get_duplicates(&self, component: &str) -> Vec<String> {
        let registry = self.registry.read().await;
        registry.get(component)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }
    
    pub async fn has_duplicates(&self, component: &str) -> bool {
        let registry = self.registry.read().await;
        registry.get(component)
            .map(|set| set.len() > 1)
            .unwrap_or(false)
    }
    
    pub async fn clear_registry(&self) {
        let mut registry = self.registry.write().await;
        registry.clear();
    }
}