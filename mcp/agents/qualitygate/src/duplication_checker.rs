//! Code Duplication Checker for QualityGate Agent
//! Detects duplicate code blocks across the codebase

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use walkdir::WalkDir;
use sha2::{Sha256, Digest};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicationReport {
    pub total_files: usize,
    pub duplicate_blocks: usize,
    pub locations: Vec<String>,
}

pub struct DuplicationChecker {
    min_block_size: usize,
    hash_cache: HashMap<String, Vec<String>>,
}

impl DuplicationChecker {
    pub fn new() -> Self {
        Self {
            min_block_size: 5, // Minimum lines for a block to be considered
            hash_cache: HashMap::new(),
        }
    }

    /// Find duplicate code blocks
    pub async fn find_duplicates(&self, workspace_path: &Path) -> Result<DuplicationReport> {
        let mut hash_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut total_files = 0;
        
        for entry in WalkDir::new(workspace_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
            .filter(|e| !e.path().to_string_lossy().contains("/target/"))
            .filter(|e| !e.path().to_string_lossy().contains("/tests/"))
        {
            total_files += 1;
            let path = entry.path();
            
            if let Ok(content) = tokio::fs::read_to_string(path).await {
                // Extract code blocks
                let blocks = self.extract_blocks(&content);
                
                for (block_hash, block_content) in blocks {
                    hash_map.entry(block_hash.clone())
                        .or_insert_with(Vec::new)
                        .push(format!("{}:{}", path.display(), block_content));
                }
            }
        }
        
        // Find duplicates (blocks appearing more than once)
        let mut duplicate_blocks = 0;
        let mut locations = Vec::new();
        
        for (hash, occurrences) in hash_map.iter() {
            if occurrences.len() > 1 {
                duplicate_blocks += 1;
                locations.push(format!(
                    "Duplicate block ({} occurrences): {}",
                    occurrences.len(),
                    occurrences.first().unwrap()
                ));
            }
        }
        
        Ok(DuplicationReport {
            total_files,
            duplicate_blocks,
            locations,
        })
    }

    /// Extract code blocks from source
    fn extract_blocks(&self, content: &str) -> Vec<(String, String)> {
        let mut blocks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        
        // Extract function bodies as blocks
        let mut in_function = false;
        let mut brace_count = 0;
        let mut current_block = Vec::new();
        let mut block_start_line = 0;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Start of function
            if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn") {
                in_function = true;
                current_block.clear();
                block_start_line = i;
            }
            
            if in_function {
                current_block.push(*line);
                
                // Count braces
                brace_count += line.chars().filter(|&c| c == '{').count();
                brace_count = brace_count.saturating_sub(line.chars().filter(|&c| c == '}').count());
                
                // End of function
                if brace_count == 0 && current_block.len() >= self.min_block_size {
                    let block_content = current_block.join("\n");
                    let normalized = self.normalize_code(&block_content);
                    let hash = self.hash_block(&normalized);
                    
                    blocks.push((hash, format!("lines {}-{}", block_start_line + 1, i + 1)));
                    
                    in_function = false;
                    current_block.clear();
                }
            }
        }
        
        blocks
    }

    /// Normalize code for comparison (remove whitespace variations)
    fn normalize_code(&self, code: &str) -> String {
        code.lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Hash a code block
    fn hash_block(&self, block: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(block.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}