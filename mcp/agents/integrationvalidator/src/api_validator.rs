//! API validation module for Bot4 integration
//! Validates API compatibility between components

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use syn::{parse_file, Item, ItemFn, ItemStruct, ReturnType, FnArg};
use regex::Regex;
use tracing::{info, debug, warn};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiEndpoint {
    pub name: String,
    pub method: String,
    pub path: String,
    pub request_type: String,
    pub response_type: String,
    pub parameters: Vec<ApiParameter>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiParameter {
    pub name: String,
    pub param_type: String,
    pub required: bool,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiContract {
    pub component: String,
    pub version: String,
    pub endpoints: Vec<ApiEndpoint>,
}

pub struct ApiValidator {
    workspace_path: PathBuf,
    api_cache: HashMap<String, ApiContract>,
}

impl ApiValidator {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self {
            workspace_path,
            api_cache: HashMap::new(),
        }
    }
    
    pub async fn validate_api_compatibility(&self, source: &str, target: &str) -> Result<bool> {
        info!("Validating API compatibility: {} -> {}", source, target);
        
        // Get APIs for both components
        let source_api = self.extract_component_api(source).await?;
        let target_api = self.extract_component_api(target).await?;
        
        // Check if source calls match target's exposed endpoints
        let compatibility = self.check_compatibility(&source_api, &target_api)?;
        
        Ok(compatibility)
    }
    
    pub async fn extract_component_api(&self, component: &str) -> Result<ApiContract> {
        let component_path = self.workspace_path.join("crates").join(component).join("src");
        let mut contract = ApiContract {
            component: component.to_string(),
            version: "1.0.0".to_string(),
            endpoints: Vec::new(),
        };
        
        // Walk through source files
        for entry in walkdir::WalkDir::new(&component_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let content = fs::read_to_string(entry.path()).await?;
            
            // Extract API endpoints from code
            contract.endpoints.extend(self.extract_endpoints(&content)?);
        }
        
        // Also check for OpenAPI/Swagger definitions
        let api_spec_path = component_path.join("api.yaml");
        if api_spec_path.exists() {
            contract.endpoints.extend(self.parse_openapi(&api_spec_path).await?);
        }
        
        Ok(contract)
    }
    
    fn extract_endpoints(&self, content: &str) -> Result<Vec<ApiEndpoint>> {
        let mut endpoints = Vec::new();
        
        // Parse Rust code for API handlers
        if let Ok(syntax_tree) = parse_file(content) {
            for item in syntax_tree.items {
                if let Item::Fn(func) = item {
                    // Look for handler functions with specific attributes
                    let is_handler = func.attrs.iter().any(|attr| {
                        let path = attr.path.segments.iter()
                            .map(|s| s.ident.to_string())
                            .collect::<Vec<_>>()
                            .join("::");
                        
                        path.contains("handler") || 
                        path.contains("post") || 
                        path.contains("get") ||
                        path.contains("put") ||
                        path.contains("delete")
                    });
                    
                    if is_handler || func.sig.ident.to_string().contains("handle") {
                        endpoints.push(self.function_to_endpoint(&func)?);
                    }
                }
            }
        }
        
        // Also look for axum/actix routes
        endpoints.extend(self.extract_web_routes(content)?);
        
        Ok(endpoints)
    }
    
    fn function_to_endpoint(&self, func: &syn::ItemFn) -> Result<ApiEndpoint> {
        let name = func.sig.ident.to_string();
        
        // Extract method from function name or attributes
        let method = if name.contains("get") {
            "GET"
        } else if name.contains("post") || name.contains("create") {
            "POST"
        } else if name.contains("put") || name.contains("update") {
            "PUT"
        } else if name.contains("delete") {
            "DELETE"
        } else {
            "POST"
        }.to_string();
        
        // Extract path from attributes or function name
        let path = format!("/api/{}", name.trim_start_matches("handle_"));
        
        // Extract request and response types
        let (request_type, response_type) = self.extract_types_from_signature(&func.sig)?;
        
        // Extract parameters
        let parameters = self.extract_parameters(&func.sig)?;
        
        Ok(ApiEndpoint {
            name,
            method,
            path,
            request_type,
            response_type,
            parameters,
        })
    }
    
    fn extract_types_from_signature(&self, sig: &syn::Signature) -> Result<(String, String)> {
        let mut request_type = "None".to_string();
        
        // Look for request type in parameters
        for arg in &sig.inputs {
            if let FnArg::Typed(pat_type) = arg {
                if let syn::Type::Path(type_path) = &*pat_type.ty {
                    let type_str = quote::quote!(#type_path).to_string();
                    if type_str.contains("Request") || type_str.contains("Json") {
                        request_type = type_str;
                        break;
                    }
                }
            }
        }
        
        // Extract response type from return type
        let response_type = match &sig.output {
            ReturnType::Default => "()".to_string(),
            ReturnType::Type(_, ty) => {
                quote::quote!(#ty).to_string()
            }
        };
        
        Ok((request_type, response_type))
    }
    
    fn extract_parameters(&self, sig: &syn::Signature) -> Result<Vec<ApiParameter>> {
        let mut parameters = Vec::new();
        
        for arg in &sig.inputs {
            if let FnArg::Typed(pat_type) = arg {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let name = pat_ident.ident.to_string();
                    
                    // Skip common parameters
                    if name == "self" || name == "ctx" || name == "state" {
                        continue;
                    }
                    
                    let param_type = quote::quote!(#pat_type.ty).to_string();
                    
                    parameters.push(ApiParameter {
                        name,
                        param_type,
                        required: true, // Assume required unless Option<T>
                        description: None,
                    });
                }
            }
        }
        
        Ok(parameters)
    }
    
    fn extract_web_routes(&self, content: &str) -> Result<Vec<ApiEndpoint>> {
        let mut endpoints = Vec::new();
        
        // Pattern for axum routes
        let axum_pattern = Regex::new(r#"\.route\("([^"]+)",\s*(\w+)::"#)?;
        for cap in axum_pattern.captures_iter(content) {
            let path = cap.get(1).unwrap().as_str().to_string();
            let method = cap.get(2).unwrap().as_str().to_uppercase();
            
            endpoints.push(ApiEndpoint {
                name: format!("route_{}", path.replace('/', "_")),
                method,
                path,
                request_type: "Unknown".to_string(),
                response_type: "Unknown".to_string(),
                parameters: Vec::new(),
            });
        }
        
        // Pattern for actix routes
        let actix_pattern = Regex::new(r#"#\[(\w+)\("([^"]+)"\)\]"#)?;
        for cap in actix_pattern.captures_iter(content) {
            let method = cap.get(1).unwrap().as_str().to_uppercase();
            let path = cap.get(2).unwrap().as_str().to_string();
            
            if ["GET", "POST", "PUT", "DELETE"].contains(&method.as_str()) {
                endpoints.push(ApiEndpoint {
                    name: format!("route_{}", path.replace('/', "_")),
                    method,
                    path,
                    request_type: "Unknown".to_string(),
                    response_type: "Unknown".to_string(),
                    parameters: Vec::new(),
                });
            }
        }
        
        Ok(endpoints)
    }
    
    async fn parse_openapi(&self, path: &PathBuf) -> Result<Vec<ApiEndpoint>> {
        let content = fs::read_to_string(path).await?;
        let mut endpoints = Vec::new();
        
        // Simple YAML parsing for OpenAPI
        if let Ok(yaml) = serde_yaml::from_str::<serde_yaml::Value>(&content) {
            if let Some(paths) = yaml["paths"].as_mapping() {
                for (path_str, methods) in paths {
                    let path = path_str.as_str().unwrap_or("").to_string();
                    
                    if let Some(methods_map) = methods.as_mapping() {
                        for (method_str, spec) in methods_map {
                            let method = method_str.as_str().unwrap_or("").to_uppercase();
                            let name = spec["operationId"].as_str()
                                .unwrap_or(&format!("{}_{}", method, path.replace('/', "_")))
                                .to_string();
                            
                            // Extract request/response from OpenAPI spec
                            let request_type = self.extract_openapi_request_type(&spec);
                            let response_type = self.extract_openapi_response_type(&spec);
                            let parameters = self.extract_openapi_parameters(&spec);
                            
                            endpoints.push(ApiEndpoint {
                                name,
                                method,
                                path: path.clone(),
                                request_type,
                                response_type,
                                parameters,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(endpoints)
    }
    
    fn extract_openapi_request_type(&self, spec: &serde_yaml::Value) -> String {
        if let Some(request_body) = &spec["requestBody"] {
            if let Some(content) = &request_body["content"]["application/json"]["schema"] {
                if let Some(ref_str) = content["$ref"].as_str() {
                    return ref_str.split('/').last().unwrap_or("Unknown").to_string();
                }
            }
        }
        "None".to_string()
    }
    
    fn extract_openapi_response_type(&self, spec: &serde_yaml::Value) -> String {
        if let Some(responses) = &spec["responses"] {
            if let Some(ok_response) = &responses["200"] {
                if let Some(content) = &ok_response["content"]["application/json"]["schema"] {
                    if let Some(ref_str) = content["$ref"].as_str() {
                        return ref_str.split('/').last().unwrap_or("Unknown").to_string();
                    }
                }
            }
        }
        "Unknown".to_string()
    }
    
    fn extract_openapi_parameters(&self, spec: &serde_yaml::Value) -> Vec<ApiParameter> {
        let mut parameters = Vec::new();
        
        if let Some(params) = spec["parameters"].as_sequence() {
            for param in params {
                parameters.push(ApiParameter {
                    name: param["name"].as_str().unwrap_or("").to_string(),
                    param_type: param["schema"]["type"].as_str().unwrap_or("string").to_string(),
                    required: param["required"].as_bool().unwrap_or(false),
                    description: param["description"].as_str().map(|s| s.to_string()),
                });
            }
        }
        
        parameters
    }
    
    fn check_compatibility(&self, source: &ApiContract, target: &ApiContract) -> Result<bool> {
        // Check if all endpoints that source expects are provided by target
        let mut compatible = true;
        
        // For simplicity, check if endpoint paths and methods match
        // In a real implementation, would also validate request/response types
        for source_endpoint in &source.endpoints {
            let found = target.endpoints.iter().any(|target_endpoint| {
                target_endpoint.path == source_endpoint.path &&
                target_endpoint.method == source_endpoint.method
            });
            
            if !found && source_endpoint.path.contains(&target.component) {
                warn!("Missing endpoint in {}: {} {}", 
                      target.component, source_endpoint.method, source_endpoint.path);
                compatible = false;
            }
        }
        
        Ok(compatible)
    }
}