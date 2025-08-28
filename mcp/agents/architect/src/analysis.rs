//! Code analysis module

use anyhow::Result;
use serde::{Deserialize, Serialize};
use syn::{parse_file, Item, ItemStruct, ItemFn, ItemTrait, ItemImpl};

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeAnalysis {
    pub structs: Vec<StructInfo>,
    pub functions: Vec<FunctionInfo>,
    pub traits: Vec<TraitInfo>,
    pub impls: Vec<ImplInfo>,
    pub complexity: ComplexityMetrics,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StructInfo {
    pub name: String,
    pub fields: usize,
    pub is_public: bool,
    pub has_generics: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub parameters: usize,
    pub is_public: bool,
    pub is_async: bool,
    pub has_generics: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TraitInfo {
    pub name: String,
    pub methods: usize,
    pub is_public: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImplInfo {
    pub trait_name: Option<String>,
    pub target_type: String,
    pub methods: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub lines_of_code: usize,
    pub cyclomatic_complexity: usize,
    pub cognitive_complexity: usize,
}

pub struct CodeAnalyzer;

impl CodeAnalyzer {
    pub fn new() -> Self {
        Self
    }
    
    pub fn analyze(&self, content: &str) -> Result<CodeAnalysis> {
        let syntax_tree = parse_file(content)?;
        
        let mut analysis = CodeAnalysis {
            structs: Vec::new(),
            functions: Vec::new(),
            traits: Vec::new(),
            impls: Vec::new(),
            complexity: ComplexityMetrics {
                lines_of_code: content.lines().count(),
                cyclomatic_complexity: 0,
                cognitive_complexity: 0,
            },
            dependencies: Vec::new(),
        };
        
        // Analyze items
        for item in syntax_tree.items {
            match item {
                Item::Struct(item_struct) => {
                    analysis.structs.push(StructInfo {
                        name: item_struct.ident.to_string(),
                        fields: item_struct.fields.len(),
                        is_public: matches!(item_struct.vis, syn::Visibility::Public(_)),
                        has_generics: !item_struct.generics.params.is_empty(),
                    });
                }
                Item::Fn(item_fn) => {
                    analysis.functions.push(FunctionInfo {
                        name: item_fn.sig.ident.to_string(),
                        parameters: item_fn.sig.inputs.len(),
                        is_public: matches!(item_fn.vis, syn::Visibility::Public(_)),
                        is_async: item_fn.sig.asyncness.is_some(),
                        has_generics: !item_fn.sig.generics.params.is_empty(),
                    });
                    
                    // Simple complexity calculation
                    analysis.complexity.cyclomatic_complexity += self.calculate_cyclomatic(&item_fn);
                }
                Item::Trait(item_trait) => {
                    analysis.traits.push(TraitInfo {
                        name: item_trait.ident.to_string(),
                        methods: item_trait.items.len(),
                        is_public: matches!(item_trait.vis, syn::Visibility::Public(_)),
                    });
                }
                Item::Impl(item_impl) => {
                    let trait_name = item_impl.trait_.as_ref().map(|(_, path, _)| {
                        path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default()
                    });
                    
                    let target_type = if let syn::Type::Path(type_path) = &*item_impl.self_ty {
                        type_path.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default()
                    } else {
                        "Unknown".to_string()
                    };
                    
                    analysis.impls.push(ImplInfo {
                        trait_name,
                        target_type,
                        methods: item_impl.items.len(),
                    });
                }
                Item::Use(item_use) => {
                    // Extract dependencies
                    if let syn::UseTree::Path(use_path) = &item_use.tree {
                        analysis.dependencies.push(use_path.ident.to_string());
                    }
                }
                _ => {}
            }
        }
        
        // Calculate cognitive complexity (simplified)
        analysis.complexity.cognitive_complexity = 
            analysis.functions.len() * 2 + 
            analysis.structs.len() + 
            analysis.traits.len() * 3;
        
        Ok(analysis)
    }
    
    fn calculate_cyclomatic(&self, func: &ItemFn) -> usize {
        // Simplified cyclomatic complexity calculation
        // Count decision points (if, match, loop, etc.)
        let mut complexity = 1; // Base complexity
        
        // This is a simplified version - real implementation would walk the AST
        let func_str = quote::quote!(#func).to_string();
        complexity += func_str.matches(" if ").count();
        complexity += func_str.matches(" match ").count();
        complexity += func_str.matches(" while ").count();
        complexity += func_str.matches(" for ").count();
        complexity += func_str.matches(" && ").count();
        complexity += func_str.matches(" || ").count();
        
        complexity
    }
}