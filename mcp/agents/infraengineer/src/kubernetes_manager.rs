//! Kubernetes deployment and management

use anyhow::{Result, bail};
use kube::{Client, Api};
use kube::api::{Patch, PatchParams, ListParams, PostParams};
use k8s_openapi::api::apps::v1::{Deployment, DeploymentSpec};
use k8s_openapi::api::core::v1::{Service, Pod};
use k8s_openapi::api::autoscaling::v1::{HorizontalPodAutoscaler, HorizontalPodAutoscalerSpec};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
pub struct ScaleResult {
    pub deployment: String,
    pub namespace: String,
    pub current_replicas: i32,
    pub target_replicas: i32,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeploymentStatus {
    pub name: String,
    pub namespace: String,
    pub replicas: i32,
    pub ready_replicas: i32,
    pub available_replicas: i32,
    pub conditions: Vec<String>,
}

pub struct KubernetesManager {
    client: Client,
    namespace: String,
}

impl KubernetesManager {
    pub async fn new() -> Result<Self> {
        // Try in-cluster config first, fallback to kubeconfig
        let client = match Client::try_default().await {
            Ok(client) => client,
            Err(_) => {
                // In development, K8s might not be available
                bail!("Kubernetes client initialization failed")
            }
        };
        
        let namespace = std::env::var("KUBERNETES_NAMESPACE")
            .unwrap_or_else(|_| "bot4".to_string());
        
        Ok(Self {
            client,
            namespace,
        })
    }
    
    pub async fn scale_deployment(&self, name: &str, replicas: u32) -> Result<ScaleResult> {
        let deployments: Api<Deployment> = Api::namespaced(self.client.clone(), &self.namespace);
        
        // Get current deployment
        let deployment = deployments.get(name).await?;
        let current_replicas = deployment.spec
            .as_ref()
            .and_then(|s| s.replicas)
            .unwrap_or(0);
        
        // Patch with new replica count
        let patch = json!({
            "spec": {
                "replicas": replicas
            }
        });
        
        let patch_params = PatchParams::apply("bot4-scaler");
        deployments.patch(name, &patch_params, &Patch::Merge(patch)).await?;
        
        Ok(ScaleResult {
            deployment: name.to_string(),
            namespace: self.namespace.clone(),
            current_replicas,
            target_replicas: replicas as i32,
            status: "scaling".to_string(),
            message: format!("Scaling {} from {} to {} replicas", name, current_replicas, replicas),
        })
    }
    
    pub async fn get_deployment_status(&self, name: &str) -> Result<DeploymentStatus> {
        let deployments: Api<Deployment> = Api::namespaced(self.client.clone(), &self.namespace);
        
        let deployment = deployments.get(name).await?;
        let status = deployment.status.ok_or_else(|| anyhow::anyhow!("No status available"))?;
        
        let conditions = status.conditions
            .unwrap_or_default()
            .iter()
            .filter(|c| c.status == "True")
            .filter_map(|c| c.type_.clone())
            .collect();
        
        Ok(DeploymentStatus {
            name: name.to_string(),
            namespace: self.namespace.clone(),
            replicas: status.replicas.unwrap_or(0),
            ready_replicas: status.ready_replicas.unwrap_or(0),
            available_replicas: status.available_replicas.unwrap_or(0),
            conditions,
        })
    }
    
    pub async fn rollout_restart(&self, deployment_name: &str) -> Result<()> {
        let deployments: Api<Deployment> = Api::namespaced(self.client.clone(), &self.namespace);
        
        // Trigger rollout by updating annotation
        let patch = json!({
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "bot4.rollout.timestamp": chrono::Utc::now().to_rfc3339()
                        }
                    }
                }
            }
        });
        
        let patch_params = PatchParams::apply("bot4-rollout");
        deployments.patch(deployment_name, &patch_params, &Patch::Merge(patch)).await?;
        
        Ok(())
    }
    
    pub async fn create_hpa(&self, deployment_name: &str, min_replicas: i32, 
                           max_replicas: i32, target_cpu_percent: i32) -> Result<()> {
        let hpas: Api<HorizontalPodAutoscaler> = Api::namespaced(self.client.clone(), &self.namespace);
        
        let hpa = HorizontalPodAutoscaler {
            metadata: k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta {
                name: Some(format!("{}-hpa", deployment_name)),
                namespace: Some(self.namespace.clone()),
                ..Default::default()
            },
            spec: Some(HorizontalPodAutoscalerSpec {
                scale_target_ref: k8s_openapi::api::autoscaling::v1::CrossVersionObjectReference {
                    api_version: Some("apps/v1".to_string()),
                    kind: "Deployment".to_string(),
                    name: deployment_name.to_string(),
                },
                min_replicas: Some(min_replicas),
                max_replicas,
                target_cpu_utilization_percentage: Some(target_cpu_percent),
            }),
            ..Default::default()
        };
        
        hpas.create(&PostParams::default(), &hpa).await?;
        
        Ok(())
    }
    
    pub async fn get_pod_logs(&self, pod_name: &str, lines: i32) -> Result<String> {
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);
        
        let log_params = kube::api::LogParams {
            tail_lines: Some(lines),
            timestamps: true,
            ..Default::default()
        };
        
        let logs = pods.logs(pod_name, &log_params).await?;
        Ok(logs)
    }
    
    pub async fn list_pods(&self, label_selector: Option<String>) -> Result<Vec<String>> {
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);
        
        let mut list_params = ListParams::default();
        if let Some(selector) = label_selector {
            list_params = list_params.labels(&selector);
        }
        
        let pod_list = pods.list(&list_params).await?;
        
        let pod_names: Vec<String> = pod_list.items
            .iter()
            .filter_map(|p| p.metadata.name.clone())
            .collect();
        
        Ok(pod_names)
    }
    
    pub async fn update_deployment_image(&self, deployment_name: &str, 
                                         container_name: &str, 
                                         new_image: &str) -> Result<()> {
        let deployments: Api<Deployment> = Api::namespaced(self.client.clone(), &self.namespace);
        
        // Get current deployment
        let mut deployment = deployments.get(deployment_name).await?;
        
        // Update container image
        if let Some(ref mut spec) = deployment.spec {
            if let Some(ref mut template) = spec.template.spec.as_mut() {
                for container in &mut template.containers {
                    if container.name == container_name {
                        container.image = Some(new_image.to_string());
                    }
                }
            }
        }
        
        // Apply update
        deployments.replace(deployment_name, &PostParams::default(), &deployment).await?;
        
        Ok(())
    }
    
    pub async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        let pods: Api<Pod> = Api::namespaced(self.client.clone(), &self.namespace);
        
        let pod_list = pods.list(&ListParams::default()).await?;
        
        let mut total_cpu_requests = 0;
        let mut total_cpu_limits = 0;
        let mut total_memory_requests = 0;
        let mut total_memory_limits = 0;
        let mut pod_count = 0;
        
        for pod in pod_list.items {
            if let Some(spec) = pod.spec {
                for container in spec.containers {
                    if let Some(resources) = container.resources {
                        // Parse resource requests
                        if let Some(requests) = resources.requests {
                            if let Some(cpu) = requests.get("cpu") {
                                total_cpu_requests += parse_cpu_value(&cpu.0);
                            }
                            if let Some(memory) = requests.get("memory") {
                                total_memory_requests += parse_memory_value(&memory.0);
                            }
                        }
                        
                        // Parse resource limits
                        if let Some(limits) = resources.limits {
                            if let Some(cpu) = limits.get("cpu") {
                                total_cpu_limits += parse_cpu_value(&cpu.0);
                            }
                            if let Some(memory) = limits.get("memory") {
                                total_memory_limits += parse_memory_value(&memory.0);
                            }
                        }
                    }
                }
                pod_count += 1;
            }
        }
        
        Ok(ResourceUsage {
            pod_count,
            total_cpu_requests_millicores: total_cpu_requests,
            total_cpu_limits_millicores: total_cpu_limits,
            total_memory_requests_mb: total_memory_requests / 1024 / 1024,
            total_memory_limits_mb: total_memory_limits / 1024 / 1024,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub pod_count: i32,
    pub total_cpu_requests_millicores: i64,
    pub total_cpu_limits_millicores: i64,
    pub total_memory_requests_mb: i64,
    pub total_memory_limits_mb: i64,
}

fn parse_cpu_value(value: &str) -> i64 {
    if value.ends_with('m') {
        value.trim_end_matches('m').parse::<i64>().unwrap_or(0)
    } else {
        // Convert CPU cores to millicores
        (value.parse::<f64>().unwrap_or(0.0) * 1000.0) as i64
    }
}

fn parse_memory_value(value: &str) -> i64 {
    if value.ends_with("Ki") {
        value.trim_end_matches("Ki").parse::<i64>().unwrap_or(0) * 1024
    } else if value.ends_with("Mi") {
        value.trim_end_matches("Mi").parse::<i64>().unwrap_or(0) * 1024 * 1024
    } else if value.ends_with("Gi") {
        value.trim_end_matches("Gi").parse::<i64>().unwrap_or(0) * 1024 * 1024 * 1024
    } else {
        value.parse::<i64>().unwrap_or(0)
    }
}