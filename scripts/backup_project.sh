#!/bin/bash
# Bot4 Project Backup Script
# Creates timestamped backups excluding build artifacts

BACKUP_DIR="/home/hamster/bot-backups"
PROJECT_DIR="/home/hamster/bot4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="bot4_backup_${TIMESTAMP}.tar.gz"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "Starting backup of Bot4 project..."
echo "Backup name: $BACKUP_NAME"

# Create the backup excluding build artifacts and git objects
cd /home/hamster
tar -czf "$BACKUP_DIR/$BACKUP_NAME" bot4/ \
    --exclude='bot4/target' \
    --exclude='bot4/rust_core/target' \
    --exclude='bot4/rust_core_old_epic7/target' \
    --exclude='bot4/rust_core/crates/*/target' \
    --exclude='bot4/rust_core/*/target' \
    --exclude='bot4/.git/objects' \
    --exclude='bot4/.git/lfs' \
    --exclude='*.log' \
    --exclude='node_modules' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache'

if [ $? -eq 0 ]; then
    SIZE=$(du -sh "$BACKUP_DIR/$BACKUP_NAME" | cut -f1)
    echo "✅ Backup completed successfully!"
    echo "📁 Location: $BACKUP_DIR/$BACKUP_NAME"
    echo "📊 Size: $SIZE"
    
    # Keep only the 5 most recent backups
    echo "Cleaning old backups..."
    cd "$BACKUP_DIR"
    ls -t bot4_backup_*.tar.gz | tail -n +6 | xargs -r rm -f
    echo "Old backups cleaned (keeping 5 most recent)"
else
    echo "❌ Backup failed!"
    exit 1
fi

# Create a backup info file
cat > "$BACKUP_DIR/backup_${TIMESTAMP}.info" << EOF
Backup Information
==================
Date: $(date)
Project: Bot4 Autonomous Trading Platform
Location: $BACKUP_DIR/$BACKUP_NAME
Size: $SIZE
Git Branch: $(cd $PROJECT_DIR && git branch --show-current)
Last Commit: $(cd $PROJECT_DIR && git log -1 --oneline)
Project Status: 13.1% Complete (508/3812 hours)

Key Components Backed Up:
- rust_core/ (main implementation)
- mathematical_ops/ (consolidated math functions)
- event_bus/ (LMAX Disruptor implementation)  
- layer_enforcement/ (architecture enforcement)
- abstractions/ (cross-layer traits)
- docs/ (architecture documentation)
- scripts/ (validation and automation)

Excluded from Backup:
- Build artifacts (target/)
- Git objects
- Log files
- Node modules
- Python cache files
EOF

echo "📝 Backup info saved to: $BACKUP_DIR/backup_${TIMESTAMP}.info"