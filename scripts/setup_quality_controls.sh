#!/bin/bash
# Setup Quality Control Systems for Bot3
# Run this IMMEDIATELY after cloning or resetting the project

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "   BOT3 QUALITY CONTROL SETUP            "
echo "=========================================="

# 1. Install Git Hooks
echo -e "\n${YELLOW}Installing Git Hooks...${NC}"
cp .git-hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
echo -e "${GREEN}✓ Pre-commit hook installed${NC}"

# 2. Make verification scripts executable
echo -e "\n${YELLOW}Setting up verification scripts...${NC}"
chmod +x scripts/verify_completion.sh
chmod +x scripts/validate_no_fakes.py
echo -e "${GREEN}✓ Verification scripts ready${NC}"

# 3. Create quality tracking directory
echo -e "\n${YELLOW}Creating quality tracking...${NC}"
mkdir -p .quality_reports
touch .quality_reports/daily_verification.log
echo -e "${GREEN}✓ Quality tracking initialized${NC}"

# 4. Install Rust quality tools
echo -e "\n${YELLOW}Installing Rust quality tools...${NC}"
cargo install cargo-tarpaulin --locked 2>/dev/null || echo "Tarpaulin already installed"
cargo install cargo-audit --locked 2>/dev/null || echo "Audit already installed"
cargo install cargo-clippy --locked 2>/dev/null || echo "Clippy already installed"
echo -e "${GREEN}✓ Rust tools ready${NC}"

# 5. Create initial quality baseline
echo -e "\n${YELLOW}Creating quality baseline...${NC}"
date > .quality_reports/baseline.txt
echo "Initial Setup Metrics:" >> .quality_reports/baseline.txt
echo "Total Rust files: $(find . -name '*.rs' | wc -l)" >> .quality_reports/baseline.txt
echo "Total lines: $(find . -name '*.rs' -exec wc -l {} + | tail -1)" >> .quality_reports/baseline.txt
echo "TODOs: $(grep -r 'TODO' --include='*.rs' . | wc -l)" >> .quality_reports/baseline.txt
echo "Unimplemented: $(grep -r 'unimplemented!' --include='*.rs' . | wc -l)" >> .quality_reports/baseline.txt
echo -e "${GREEN}✓ Baseline recorded${NC}"

# 6. Run initial verification
echo -e "\n${YELLOW}Running initial quality check...${NC}"
./scripts/verify_completion.sh || true

# 7. Create daily cron job (optional)
echo -e "\n${YELLOW}Setting up daily verification (optional)...${NC}"
CRON_CMD="0 9 * * * cd /home/hamster/bot4 && ./scripts/verify_completion.sh >> .quality_reports/daily_verification.log 2>&1"
(crontab -l 2>/dev/null | grep -v "verify_completion.sh" ; echo "$CRON_CMD") | crontab -

echo -e "\n=========================================="
echo -e "${GREEN}✓✓✓ QUALITY CONTROLS INSTALLED ✓✓✓${NC}"
echo -e "=========================================="
echo ""
echo "Quality enforcement is now active:"
echo "1. Pre-commit hooks will prevent fake implementations"
echo "2. Run './scripts/verify_completion.sh' before marking tasks complete"
echo "3. Daily verification runs at 9 AM"
echo "4. See .quality_reports/ for tracking"
echo ""
echo -e "${YELLOW}REMEMBER: No task is complete until verification passes!${NC}"