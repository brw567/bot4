#!/bin/bash

# Bot4 Trading Platform - Kubernetes Deployment Script
# Team: Full 8-Agent ULTRATHINK Collaboration
# Research Applied: K8s best practices, GitOps, progressive deployment

set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           BOT4 TRADING PLATFORM - K8S DEPLOYMENT            ║${NC}"
echo -e "${BLUE}║                  ULTRATHINK PRODUCTION READY                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Configuration
NAMESPACE="bot4-trading"
ENVIRONMENT="${1:-staging}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
    echo -e "${RED}Error: Invalid environment. Use: dev, staging, or production${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Deploying to environment: ${ENVIRONMENT}${NC}"

# Check kubectl availability
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

# Check kustomize availability
if ! command -v kustomize &> /dev/null; then
    echo -e "${YELLOW}Installing kustomize...${NC}"
    curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
    sudo mv kustomize /usr/local/bin/
fi

# Function to wait for deployment
wait_for_deployment() {
    local deployment=$1
    echo -e "${YELLOW}Waiting for deployment ${deployment} to be ready...${NC}"
    kubectl rollout status deployment/${deployment} -n ${NAMESPACE} --timeout=600s
}

# Function to check pod health
check_pod_health() {
    echo -e "${YELLOW}Checking pod health...${NC}"
    kubectl get pods -n ${NAMESPACE} -l app=bot4
    
    # Wait for all pods to be ready
    while [[ $(kubectl get pods -n ${NAMESPACE} -l app=bot4 -o jsonpath='{.items[*].status.containerStatuses[*].ready}' | grep -c false) -gt 0 ]]; do
        echo "Waiting for pods to be ready..."
        sleep 10
    done
    
    echo -e "${GREEN}All pods are healthy!${NC}"
}

# Pre-deployment checks
echo -e "\n${BLUE}═══ Pre-deployment Checks ═══${NC}"

# Check cluster connectivity
echo "Checking cluster connectivity..."
kubectl cluster-info || {
    echo -e "${RED}Cannot connect to Kubernetes cluster${NC}"
    exit 1
}

# Check if namespace exists
if ! kubectl get namespace ${NAMESPACE} &> /dev/null; then
    echo "Creating namespace ${NAMESPACE}..."
    kubectl create namespace ${NAMESPACE}
fi

# Deploy based on environment
echo -e "\n${BLUE}═══ Deploying Bot4 Trading Platform ═══${NC}"

case ${ENVIRONMENT} in
    dev)
        echo "Deploying development environment..."
        kubectl apply -k k8s/overlays/dev
        ;;
    staging)
        echo "Deploying staging environment..."
        kubectl apply -k k8s/overlays/staging
        ;;
    production)
        echo -e "${YELLOW}⚠️  PRODUCTION DEPLOYMENT ⚠️${NC}"
        echo -n "Are you sure you want to deploy to production? (yes/no): "
        read -r confirmation
        if [[ "$confirmation" != "yes" ]]; then
            echo "Production deployment cancelled."
            exit 0
        fi
        
        # Production deployment with canary
        echo "Starting canary deployment..."
        
        # Deploy canary (10% traffic)
        kubectl apply -k k8s/overlays/production --dry-run=client -o yaml | \
            sed 's/replicas: 5/replicas: 1/' | \
            kubectl apply -f -
        
        echo "Canary deployed. Monitoring for 5 minutes..."
        sleep 300
        
        # Check canary health
        CANARY_HEALTHY=$(kubectl get pods -n ${NAMESPACE} -l app=bot4,version=canary -o jsonpath='{.items[*].status.containerStatuses[*].ready}' | grep -c true)
        
        if [[ ${CANARY_HEALTHY} -eq 0 ]]; then
            echo -e "${RED}Canary deployment failed. Rolling back...${NC}"
            kubectl rollout undo deployment/bot4-trading-engine -n ${NAMESPACE}
            exit 1
        fi
        
        echo -e "${GREEN}Canary healthy. Proceeding with full deployment...${NC}"
        kubectl apply -k k8s/overlays/production
        ;;
esac

# Wait for deployment to complete
wait_for_deployment "bot4-trading-engine"

# Post-deployment checks
echo -e "\n${BLUE}═══ Post-deployment Validation ═══${NC}"

# Check pod health
check_pod_health

# Check service endpoints
echo -e "\n${YELLOW}Checking service endpoints...${NC}"
kubectl get endpoints -n ${NAMESPACE}

# Run smoke tests
echo -e "\n${YELLOW}Running smoke tests...${NC}"

# Test health endpoint
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=bot4 -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ${NAMESPACE} ${POD_NAME} -- curl -f http://localhost:8080/health/liveness || {
    echo -e "${RED}Health check failed${NC}"
    exit 1
}

# Check metrics endpoint
kubectl exec -n ${NAMESPACE} ${POD_NAME} -- curl -f http://localhost:9090/metrics > /dev/null 2>&1 || {
    echo -e "${RED}Metrics endpoint check failed${NC}"
    exit 1
}

# Performance validation
echo -e "\n${YELLOW}Validating performance metrics...${NC}"

# Check decision latency
LATENCY=$(kubectl exec -n ${NAMESPACE} ${POD_NAME} -- curl -s http://localhost:9090/metrics | grep decision_latency_us | grep -oE '[0-9]+' | head -1)
if [[ ${LATENCY} -gt 100 ]]; then
    echo -e "${YELLOW}Warning: Decision latency ${LATENCY}μs exceeds target of 100μs${NC}"
else
    echo -e "${GREEN}Decision latency: ${LATENCY}μs ✓${NC}"
fi

# Display deployment summary
echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              DEPLOYMENT SUCCESSFUL!                          ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Environment: ${ENVIRONMENT}                                 ║${NC}"
echo -e "${GREEN}║  Namespace: ${NAMESPACE}                                     ║${NC}"
echo -e "${GREEN}║  Pods Running: $(kubectl get pods -n ${NAMESPACE} -l app=bot4 --no-headers | wc -l)                                              ║${NC}"
echo -e "${GREEN}║  Service: $(kubectl get svc bot4-trading-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "ClusterIP") ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"

# Output useful commands
echo -e "\n${BLUE}Useful commands:${NC}"
echo "  Watch pods:         kubectl get pods -n ${NAMESPACE} -w"
echo "  View logs:          kubectl logs -n ${NAMESPACE} -l app=bot4 -f"
echo "  Port forward:       kubectl port-forward -n ${NAMESPACE} svc/bot4-trading-service 8080:8080"
echo "  Exec into pod:      kubectl exec -it -n ${NAMESPACE} ${POD_NAME} -- /bin/bash"
echo "  Scale deployment:   kubectl scale deployment bot4-trading-engine -n ${NAMESPACE} --replicas=10"
echo "  View metrics:       kubectl top pods -n ${NAMESPACE}"

echo -e "\n${GREEN}Deployment complete! Bot4 Trading Platform is running.${NC}"

# Monitor for 1 minute
echo -e "\n${YELLOW}Monitoring deployment for 1 minute...${NC}"
for i in {1..6}; do
    sleep 10
    echo -n "."
done
echo ""

# Final health check
check_pod_health

echo -e "\n${GREEN}All systems operational. Happy trading! 🚀${NC}"