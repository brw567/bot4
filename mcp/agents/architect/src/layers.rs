//! Layer architecture enforcement

use anyhow::{Result, bail};
use std::collections::HashMap;

pub struct LayerEnforcer {
    layer_hierarchy: HashMap<String, u8>,
}

impl LayerEnforcer {
    pub fn new() -> Self {
        let mut hierarchy = HashMap::new();
        hierarchy.insert("infrastructure".to_string(), 0);
        hierarchy.insert("data_ingestion".to_string(), 1);
        hierarchy.insert("data".to_string(), 1); // Alias
        hierarchy.insert("risk".to_string(), 2);
        hierarchy.insert("ml".to_string(), 3);
        hierarchy.insert("strategies".to_string(), 4);
        hierarchy.insert("trading_engine".to_string(), 5);
        hierarchy.insert("execution".to_string(), 5); // Alias
        hierarchy.insert("integration".to_string(), 6);
        
        Self {
            layer_hierarchy: hierarchy,
        }
    }
    
    pub fn check_violation(&self, source_layer: &str, target_layer: &str) -> Result<bool> {
        let source_level = self.layer_hierarchy.get(source_layer)
            .ok_or_else(|| anyhow::anyhow!("Unknown source layer: {}", source_layer))?;
        
        let target_level = self.layer_hierarchy.get(target_layer)
            .ok_or_else(|| anyhow::anyhow!("Unknown target layer: {}", target_layer))?;
        
        // Violation if target level is higher than source level
        Ok(target_level > source_level)
    }
    
    pub fn get_allowed_dependencies(&self, layer: &str) -> Result<Vec<String>> {
        let level = self.layer_hierarchy.get(layer)
            .ok_or_else(|| anyhow::anyhow!("Unknown layer: {}", layer))?;
        
        let mut allowed = Vec::new();
        for (name, layer_level) in &self.layer_hierarchy {
            if layer_level <= level {
                allowed.push(name.clone());
            }
        }
        
        allowed.sort();
        Ok(allowed)
    }
    
    pub fn validate_import(&self, source_file: &str, import_path: &str) -> Result<bool> {
        // Extract layer from file path
        let source_layer = self.extract_layer_from_path(source_file)?;
        let target_layer = self.extract_layer_from_import(import_path)?;
        
        self.check_violation(&source_layer, &target_layer)
    }
    
    fn extract_layer_from_path(&self, path: &str) -> Result<String> {
        // Parse path to determine layer
        for layer in self.layer_hierarchy.keys() {
            if path.contains(layer) {
                return Ok(layer.clone());
            }
        }
        bail!("Could not determine layer from path: {}", path)
    }
    
    fn extract_layer_from_import(&self, import: &str) -> Result<String> {
        // Parse import to determine target layer
        for layer in self.layer_hierarchy.keys() {
            if import.contains(layer) {
                return Ok(layer.clone());
            }
        }
        bail!("Could not determine layer from import: {}", import)
    }
}