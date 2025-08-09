#!/bin/bash
# Enhanced Remote Deployment Script for Bot3
# Jordan's production deployment tool

set -e

# Configuration
REMOTE_HOST="192.168.100.64"
REMOTE_USER="hamster"
REMOTE_PATH="/home/hamster/bot3_production"
BACKUP_PATH="/home/hamster/backups"
PROJECT_NAME="bot3"
DOCKER_IMAGE="bot3-trading"
VERSION_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Timestamp for backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Pre-deployment validation
pre_deploy_checks() {
    log_step "Running pre-deployment checks..."
    
    # Check Python tests
    log_info "Running unit tests..."
    if ! pytest tests/unit/ -q; then
        log_error "Unit tests failed!"
        exit 1
    fi
    
    # Validate no fake implementations (Sam's check)
    log_info "Checking for fake implementations..."
    if ! python scripts/validate_no_fakes.py; then
        log_error "Fake implementations detected! Sam rejects this deployment."
        exit 1
    fi
    
    # Check risk limits (Quinn's check)
    log_info "Validating risk limits..."
    if ! python scripts/check_risk_limits.py; then
        log_error "Risk limits exceeded! Quinn vetoes this deployment."
        exit 1
    fi
    
    # Check Docker build
    log_info "Testing Docker build..."
    if ! docker build -t ${DOCKER_IMAGE}:test . > /dev/null 2>&1; then
        log_error "Docker build failed!"
        exit 1
    fi
    
    log_info "✅ All pre-deployment checks passed!"
}

# Create backup of current deployment
backup_current() {
    log_step "Creating backup of current deployment..."
    
    ssh ${REMOTE_USER}@${REMOTE_HOST} "
        if [ -d ${REMOTE_PATH} ]; then
            mkdir -p ${BACKUP_PATH}
            cd ${REMOTE_PATH}
            
            # Stop services gracefully
            docker-compose down --timeout 30 2>/dev/null || true
            
            # Create backup
            tar -czf ${BACKUP_PATH}/${PROJECT_NAME}_${TIMESTAMP}.tar.gz \
                --exclude='*.log' \
                --exclude='__pycache__' \
                --exclude='data/historical/*' \
                .
            
            echo 'Backup created: ${PROJECT_NAME}_${TIMESTAMP}.tar.gz'
        else
            echo 'No existing deployment to backup'
        fi
    "
    
    log_info "Backup completed"
}

# Build and push Docker images
build_and_push() {
    log_step "Building Docker images..."
    
    # Build with version tag
    VERSION_TAG="${VERSION_TAG:-$(git rev-parse --short HEAD)}"
    
    docker build \
        -t ${DOCKER_IMAGE}:${VERSION_TAG} \
        -t ${DOCKER_IMAGE}:latest \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VERSION=${VERSION_TAG} \
        .
    
    # Save image for transfer
    log_info "Saving Docker image..."
    docker save ${DOCKER_IMAGE}:${VERSION_TAG} | gzip > ${DOCKER_IMAGE}.tar.gz
    
    log_info "Docker image built: ${DOCKER_IMAGE}:${VERSION_TAG}"
}

# Deploy to remote host
deploy_to_remote() {
    log_step "Deploying to remote host ${REMOTE_HOST}..."
    
    # Transfer Docker image
    log_info "Transferring Docker image..."
    scp ${DOCKER_IMAGE}.tar.gz ${REMOTE_USER}@${REMOTE_HOST}:/tmp/
    
    # Transfer application files
    log_info "Transferring application files..."
    rsync -avz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='venv' \
        --exclude='data/historical' \
        --exclude='.env' \
        . ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
    
    # Deploy on remote
    log_info "Starting deployment on remote..."
    ssh ${REMOTE_USER}@${REMOTE_HOST} "
        cd ${REMOTE_PATH}
        
        # Load Docker image
        echo 'Loading Docker image...'
        docker load < /tmp/${DOCKER_IMAGE}.tar.gz
        rm /tmp/${DOCKER_IMAGE}.tar.gz
        
        # Update environment file if needed
        if [ ! -f .env ]; then
            cp .env.example .env
            echo 'Please configure .env file on remote!'
        fi
        
        # Start services
        echo 'Starting services...'
        docker-compose pull
        docker-compose up -d --remove-orphans
        
        # Clean up old images
        docker image prune -f
    "
    
    log_info "Deployment completed"
}

# Health check
health_check() {
    log_step "Running health checks..."
    
    # Wait for services to start
    log_info "Waiting for services to start..."
    sleep 10
    
    # Check if containers are running
    log_info "Checking Docker containers..."
    ssh ${REMOTE_USER}@${REMOTE_HOST} "
        docker ps | grep ${PROJECT_NAME} || exit 1
    " || {
        log_error "Docker containers not running!"
        return 1
    }
    
    # Check API health endpoint
    log_info "Checking API health..."
    for i in {1..5}; do
        if curl -f -s http://${REMOTE_HOST}:8000/health > /dev/null; then
            log_info "✅ API is healthy"
            break
        else
            if [ $i -eq 5 ]; then
                log_error "API health check failed!"
                return 1
            fi
            log_warn "API not ready, retrying in 5 seconds..."
            sleep 5
        fi
    done
    
    # Check database connection
    log_info "Checking database connection..."
    ssh ${REMOTE_USER}@${REMOTE_HOST} "
        docker exec ${PROJECT_NAME}_postgres_1 pg_isready || exit 1
    " || {
        log_error "Database not responding!"
        return 1
    }
    
    # Check Redis
    log_info "Checking Redis..."
    ssh ${REMOTE_USER}@${REMOTE_HOST} "
        docker exec ${PROJECT_NAME}_redis_1 redis-cli ping || exit 1
    " || {
        log_error "Redis not responding!"
        return 1
    }
    
    log_info "✅ All health checks passed!"
}

# Rollback to previous version
rollback() {
    log_step "Rolling back to previous version..."
    
    ssh ${REMOTE_USER}@${REMOTE_HOST} "
        cd ${BACKUP_PATH}
        
        # Find latest backup
        LATEST_BACKUP=\$(ls -t ${PROJECT_NAME}_*.tar.gz 2>/dev/null | head -1)
        
        if [ -z \"\$LATEST_BACKUP\" ]; then
            echo 'No backup found to rollback to!'
            exit 1
        fi
        
        echo \"Rolling back to: \$LATEST_BACKUP\"
        
        # Stop current deployment
        cd ${REMOTE_PATH}
        docker-compose down --timeout 30
        
        # Restore backup
        tar -xzf ${BACKUP_PATH}/\$LATEST_BACKUP -C ${REMOTE_PATH}/
        
        # Start services
        docker-compose up -d
    "
    
    log_info "Rollback completed"
}

# Show deployment status
show_status() {
    log_step "Deployment Status"
    
    ssh ${REMOTE_USER}@${REMOTE_HOST} "
        echo '=== Docker Containers ==='
        docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep ${PROJECT_NAME} || echo 'No containers running'
        
        echo -e '\n=== Recent Logs ==='
        docker logs ${PROJECT_NAME}_trading_1 --tail 10 2>/dev/null || echo 'No logs available'
        
        echo -e '\n=== System Resources ==='
        docker stats --no-stream --format 'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}' | grep ${PROJECT_NAME} || true
    "
}

# Main execution
main() {
    case "${1:-}" in
        --fast)
            log_info "Fast deployment (skipping checks)..."
            build_and_push
            deploy_to_remote
            health_check
            ;;
            
        --backup)
            backup_current
            if [ "${2:-}" == "--deploy" ]; then
                pre_deploy_checks
                build_and_push
                deploy_to_remote
                health_check
            fi
            ;;
            
        --rollback)
            rollback
            health_check
            ;;
            
        --status)
            show_status
            ;;
            
        --check)
            pre_deploy_checks
            log_info "All checks passed, ready to deploy!"
            ;;
            
        --help)
            echo "Usage: $0 [option]"
            echo "Options:"
            echo "  (no option)  - Full deployment with all checks"
            echo "  --fast       - Skip pre-deployment checks"
            echo "  --backup     - Create backup only"
            echo "  --backup --deploy - Backup then deploy"
            echo "  --rollback   - Rollback to previous version"
            echo "  --status     - Show deployment status"
            echo "  --check      - Run pre-deployment checks only"
            echo "  --help       - Show this help message"
            ;;
            
        *)
            # Full deployment process
            log_info "Starting full deployment process..."
            pre_deploy_checks
            backup_current
            build_and_push
            deploy_to_remote
            
            if health_check; then
                log_info "🚀 Deployment successful!"
                show_status
            else
                log_error "Deployment failed! Rolling back..."
                rollback
                exit 1
            fi
            ;;
    esac
}

# Execute main function
main "$@"