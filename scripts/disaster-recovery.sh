#!/bin/bash

# Bot3 Trading Platform - Disaster Recovery Script
# Provides backup, restore, and failover capabilities

set -euo pipefail

# Configuration
NAMESPACE="bot3-trading"
BACKUP_BUCKET="s3://bot3-backups"
REGIONS=("us-east-1" "eu-west-1" "ap-southeast-1")
BACKUP_RETENTION_DAYS=30
RTO_TARGET_MINUTES=5
RPO_TARGET_MINUTES=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    commands=("kubectl" "aws" "velero" "pg_dump" "redis-cli")
    for cmd in "${commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "$cmd is required but not installed"
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    log "Prerequisites check passed"
}

# Backup functions
backup_databases() {
    log "Backing up databases..."
    
    # PostgreSQL backup
    POD=$(kubectl get pod -n $NAMESPACE -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n $NAMESPACE $POD -- pg_dumpall -U bot3user | \
        gzip | \
        aws s3 cp - "$BACKUP_BUCKET/postgres/backup-$(date +%Y%m%d-%H%M%S).sql.gz"
    
    # Redis backup
    POD=$(kubectl get pod -n $NAMESPACE -l app=redis -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n $NAMESPACE $POD -- redis-cli BGSAVE
    sleep 5
    kubectl cp $NAMESPACE/$POD:/data/dump.rdb - | \
        gzip | \
        aws s3 cp - "$BACKUP_BUCKET/redis/backup-$(date +%Y%m%d-%H%M%S).rdb.gz"
    
    log "Database backup completed"
}

backup_persistent_volumes() {
    log "Backing up persistent volumes..."
    
    # Use Velero for PV backup
    velero backup create "bot3-backup-$(date +%Y%m%d-%H%M%S)" \
        --include-namespaces $NAMESPACE \
        --include-resources persistentvolumeclaims,persistentvolumes \
        --storage-location default \
        --wait
    
    log "Persistent volume backup completed"
}

backup_configuration() {
    log "Backing up configuration..."
    
    # Export all Kubernetes resources
    kubectl get all,cm,secret,pvc,ingress -n $NAMESPACE -o yaml | \
        gzip | \
        aws s3 cp - "$BACKUP_BUCKET/k8s/config-$(date +%Y%m%d-%H%M%S).yaml.gz"
    
    # Backup Helm values
    helm get values bot3-trading -n $NAMESPACE | \
        aws s3 cp - "$BACKUP_BUCKET/helm/values-$(date +%Y%m%d-%H%M%S).yaml"
    
    log "Configuration backup completed"
}

backup_application_state() {
    log "Backing up application state..."
    
    # Get trading engine state
    kubectl exec -n $NAMESPACE deploy/bot3-trading-engine -- \
        /usr/local/bin/bot3-export-state | \
        gzip | \
        aws s3 cp - "$BACKUP_BUCKET/state/trading-$(date +%Y%m%d-%H%M%S).json.gz"
    
    # Export ML models
    kubectl cp $NAMESPACE/bot3-trading-engine:/models - | \
        tar -czf - | \
        aws s3 cp - "$BACKUP_BUCKET/models/models-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    log "Application state backup completed"
}

# Restore functions
restore_databases() {
    local BACKUP_DATE=$1
    log "Restoring databases from $BACKUP_DATE..."
    
    # PostgreSQL restore
    aws s3 cp "$BACKUP_BUCKET/postgres/backup-$BACKUP_DATE.sql.gz" - | \
        gunzip | \
        kubectl exec -i -n $NAMESPACE deploy/postgres -- psql -U bot3user
    
    # Redis restore
    POD=$(kubectl get pod -n $NAMESPACE -l app=redis -o jsonpath='{.items[0].metadata.name}')
    aws s3 cp "$BACKUP_BUCKET/redis/backup-$BACKUP_DATE.rdb.gz" - | \
        gunzip | \
        kubectl cp - $NAMESPACE/$POD:/data/dump.rdb
    kubectl exec -n $NAMESPACE $POD -- redis-cli SHUTDOWN NOSAVE
    kubectl delete pod -n $NAMESPACE $POD
    
    log "Database restore completed"
}

restore_persistent_volumes() {
    local BACKUP_NAME=$1
    log "Restoring persistent volumes..."
    
    velero restore create --from-backup $BACKUP_NAME --wait
    
    log "Persistent volume restore completed"
}

restore_configuration() {
    local BACKUP_DATE=$1
    log "Restoring configuration..."
    
    aws s3 cp "$BACKUP_BUCKET/k8s/config-$BACKUP_DATE.yaml.gz" - | \
        gunzip | \
        kubectl apply -f -
    
    log "Configuration restore completed"
}

# Failover functions
initiate_failover() {
    local TARGET_REGION=$1
    log "Initiating failover to $TARGET_REGION..."
    
    # Update DNS to point to new region
    aws route53 change-resource-record-sets \
        --hosted-zone-id Z123456789 \
        --change-batch '{
            "Changes": [{
                "Action": "UPSERT",
                "ResourceRecordSet": {
                    "Name": "api.bot3.ai",
                    "Type": "A",
                    "AliasTarget": {
                        "HostedZoneId": "Z215JYRZR8TBV5",
                        "DNSName": "'"$TARGET_REGION"'.elb.amazonaws.com",
                        "EvaluateTargetHealth": false
                    }
                }
            }]
        }'
    
    # Scale up in target region
    kubectl config use-context "bot3-$TARGET_REGION"
    kubectl scale deploy bot3-trading-engine -n $NAMESPACE --replicas=20
    
    # Verify health
    sleep 30
    if ! check_health; then
        error "Failover health check failed"
    fi
    
    log "Failover to $TARGET_REGION completed"
}

failback() {
    local ORIGINAL_REGION=$1
    log "Failing back to $ORIGINAL_REGION..."
    
    # Ensure original region is healthy
    kubectl config use-context "bot3-$ORIGINAL_REGION"
    if ! check_health; then
        error "Original region not healthy, cannot failback"
    fi
    
    # Sync data from DR region
    sync_data_between_regions $TARGET_REGION $ORIGINAL_REGION
    
    # Update DNS
    initiate_failover $ORIGINAL_REGION
    
    # Scale down DR region
    kubectl config use-context "bot3-$TARGET_REGION"
    kubectl scale deploy bot3-trading-engine -n $NAMESPACE --replicas=2
    
    log "Failback completed"
}

# Health check
check_health() {
    log "Checking system health..."
    
    # Check deployment status
    if ! kubectl rollout status deploy/bot3-trading-engine -n $NAMESPACE --timeout=60s; then
        return 1
    fi
    
    # Check service endpoints
    ENDPOINT=$(kubectl get svc bot3-trading-engine -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if ! curl -sf "http://$ENDPOINT:8080/health" > /dev/null; then
        return 1
    fi
    
    # Check database connectivity
    if ! kubectl exec -n $NAMESPACE deploy/bot3-trading-engine -- /usr/local/bin/bot3-health-check; then
        return 1
    fi
    
    log "Health check passed"
    return 0
}

# Continuous backup
continuous_backup() {
    log "Starting continuous backup..."
    
    while true; do
        backup_databases
        backup_application_state
        
        # Clean old backups
        aws s3 ls "$BACKUP_BUCKET/" --recursive | \
            awk '{print $4}' | \
            while read -r file; do
                FILE_DATE=$(echo $file | grep -oP '\d{8}' | head -1)
                if [[ $(date -d "$FILE_DATE" +%s) -lt $(date -d "$BACKUP_RETENTION_DAYS days ago" +%s) ]]; then
                    aws s3 rm "$BACKUP_BUCKET/$file"
                    log "Deleted old backup: $file"
                fi
            done
        
        sleep 60  # Backup every minute for RPO < 1 minute
    done
}

# Disaster recovery test
dr_test() {
    log "Starting disaster recovery test..."
    
    # Create test namespace
    kubectl create namespace bot3-dr-test || true
    
    # Restore latest backup to test namespace
    LATEST_BACKUP=$(aws s3 ls "$BACKUP_BUCKET/postgres/" | sort | tail -1 | awk '{print $4}')
    
    # Measure RTO
    START_TIME=$(date +%s)
    
    restore_databases "${LATEST_BACKUP%.sql.gz}"
    restore_configuration "${LATEST_BACKUP%.sql.gz}"
    
    END_TIME=$(date +%s)
    RTO=$((END_TIME - START_TIME))
    
    log "DR test completed. RTO: ${RTO} seconds"
    
    if [[ $RTO -gt $((RTO_TARGET_MINUTES * 60)) ]]; then
        warning "RTO target not met: ${RTO}s > ${RTO_TARGET_MINUTES}m"
    fi
    
    # Cleanup
    kubectl delete namespace bot3-dr-test
}

# Monitor backup status
monitor_backups() {
    log "Monitoring backup status..."
    
    # Check last backup time
    LAST_BACKUP=$(aws s3 ls "$BACKUP_BUCKET/" --recursive | sort | tail -1 | awk '{print $1, $2}')
    LAST_BACKUP_EPOCH=$(date -d "$LAST_BACKUP" +%s)
    CURRENT_EPOCH=$(date +%s)
    MINUTES_SINCE_BACKUP=$(((CURRENT_EPOCH - LAST_BACKUP_EPOCH) / 60))
    
    if [[ $MINUTES_SINCE_BACKUP -gt $RPO_TARGET_MINUTES ]]; then
        error "RPO breach: No backup for ${MINUTES_SINCE_BACKUP} minutes"
    fi
    
    # Check backup integrity
    aws s3 ls "$BACKUP_BUCKET/" --recursive | \
        awk '{print $4}' | \
        while read -r file; do
            if ! aws s3api head-object --bucket bot3-backups --key "$file" &> /dev/null; then
                warning "Backup file corrupted: $file"
            fi
        done
    
    log "Backup monitoring completed"
}

# Main menu
show_menu() {
    echo "Bot3 Disaster Recovery System"
    echo "=============================="
    echo "1. Perform full backup"
    echo "2. Restore from backup"
    echo "3. Initiate failover"
    echo "4. Failback to primary"
    echo "5. Test disaster recovery"
    echo "6. Start continuous backup"
    echo "7. Monitor backup status"
    echo "8. Check system health"
    echo "9. Exit"
    echo ""
    read -p "Select option: " choice
}

# Main execution
main() {
    check_prerequisites
    
    case "${1:-menu}" in
        backup)
            backup_databases
            backup_persistent_volumes
            backup_configuration
            backup_application_state
            ;;
        restore)
            restore_databases "$2"
            restore_persistent_volumes "$2"
            restore_configuration "$2"
            ;;
        failover)
            initiate_failover "$2"
            ;;
        failback)
            failback "$2"
            ;;
        test)
            dr_test
            ;;
        continuous)
            continuous_backup
            ;;
        monitor)
            monitor_backups
            ;;
        health)
            check_health
            ;;
        menu|*)
            while true; do
                show_menu
                case $choice in
                    1) main backup ;;
                    2) read -p "Backup date (YYYYMMDD-HHMMSS): " date && main restore "$date" ;;
                    3) read -p "Target region: " region && main failover "$region" ;;
                    4) read -p "Original region: " region && main failback "$region" ;;
                    5) main test ;;
                    6) main continuous ;;
                    7) main monitor ;;
                    8) main health ;;
                    9) exit 0 ;;
                    *) echo "Invalid option" ;;
                esac
                echo ""
                read -p "Press Enter to continue..."
            done
            ;;
    esac
}

# Run main function
main "$@"