#!/bin/bash
# ULTRATHINK K8s Deployment Script
# Team: Full 8-Agent Collaboration

set -e

ENVIRONMENT=${1:-staging}
NAMESPACE="bot4-trading"

echo "🚀 Deploying Bot4 to Kubernetes ($ENVIRONMENT)"
echo "==========================================="

# Check kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster."
    exit 1
fi

echo "✅ Connected to cluster: $(kubectl config current-context)"

# Apply base configuration
echo "📦 Applying base configuration..."
kubectl apply -k k8s/base/

# Apply environment overlay
echo "🔧 Applying $ENVIRONMENT overlay..."
kubectl apply -k k8s/overlays/$ENVIRONMENT/

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/bot4-trading-engine -n $NAMESPACE --timeout=600s

# Check pod status
echo "📊 Pod Status:"
kubectl get pods -n $NAMESPACE -l app=bot4

# Show services
echo "🌐 Services:"
kubectl get svc -n $NAMESPACE

# Show HPA status
echo "📈 Autoscaling Status:"
kubectl get hpa -n $NAMESPACE

# Show PDB status
echo "🛡️ PodDisruptionBudget Status:"
kubectl get pdb -n $NAMESPACE

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "Access points:"
echo "  - Metrics: http://$(kubectl get node -o wide | grep Ready | head -1 | awk '{print $6}'):30090/metrics"
echo "  - Health: kubectl port-forward -n $NAMESPACE svc/bot4-trading-service 8080:8080"
echo ""
echo "Useful commands:"
echo "  - Logs: kubectl logs -n $NAMESPACE -l app=bot4 --tail=100 -f"
echo "  - Shell: kubectl exec -it -n $NAMESPACE deploy/bot4-trading-engine -- /bin/sh"
echo "  - Scale: kubectl scale deploy/bot4-trading-engine -n $NAMESPACE --replicas=20"
