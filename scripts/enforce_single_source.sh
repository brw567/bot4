#!/bin/bash
# ENFORCE SINGLE SOURCE OF TRUTH POLICY
# Karl's Project Management Enforcement Tool
# Date: 2025-08-27
# Version: 1.0

set -e

echo "=========================================="
echo "   SINGLE SOURCE OF TRUTH ENFORCEMENT    "
echo "   Project Manager: Karl                 "
echo "=========================================="
echo ""

# Define canonical documents
CANONICAL_DOCS=(
    "PROJECT_MANAGEMENT_MASTER.md"
    "docs/MASTER_ARCHITECTURE.md"
    "docs/LLM_OPTIMIZED_ARCHITECTURE.md"
    "CLAUDE.md"
    "README.md"
)

# Check for violations
VIOLATIONS=0

echo "🔍 Checking for duplicate documentation..."
echo ""

# Check for versioned files (excluding archived folders)
VERSIONED=$(find . -type f \( -name "*_v[0-9]*.md" -o -name "*_V[0-9]*.md" -o -name "*.backup*" \) 2>/dev/null | grep -v ".git" | grep -v "archived_plans" | grep -v "archived_docs" | grep -v ".archive_old" | grep -v "_archived_" || true)

if [ ! -z "$VERSIONED" ]; then
    echo "❌ VIOLATION: Versioned documentation found:"
    echo "$VERSIONED" | while read file; do
        echo "   - $file"
    done
    echo ""
    VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check for duplicate project management docs
PM_DOCS=$(find . -type f -name "*PROJECT_MANAGEMENT*.md" 2>/dev/null | grep -v "./PROJECT_MANAGEMENT_MASTER.md" | grep -v ".git" | grep -v "archived_plans" || true)

if [ ! -z "$PM_DOCS" ]; then
    echo "❌ VIOLATION: Duplicate project management docs found:"
    echo "$PM_DOCS" | while read file; do
        echo "   - $file"
    done
    echo ""
    VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check for duplicate architecture docs
ARCH_DOCS=$(find ./docs -type f -name "*ARCHITECTURE*.md" 2>/dev/null | grep -v "./docs/MASTER_ARCHITECTURE.md" | grep -v "./docs/LLM_OPTIMIZED_ARCHITECTURE.md" | grep -v ".git" || true)

if [ ! -z "$ARCH_DOCS" ]; then
    echo "❌ VIOLATION: Duplicate architecture docs found:"
    echo "$ARCH_DOCS" | while read file; do
        echo "   - $file"
    done
    echo ""
    VIOLATIONS=$((VIOLATIONS + 1))
fi

# Check canonical documents exist
echo "📋 Verifying canonical documents..."
MISSING=0
for doc in "${CANONICAL_DOCS[@]}"; do
    if [ ! -f "$doc" ]; then
        echo "   ❌ Missing: $doc"
        MISSING=$((MISSING + 1))
    else
        echo "   ✅ Found: $doc"
    fi
done
echo ""

# Report results
echo "=========================================="
if [ $VIOLATIONS -eq 0 ] && [ $MISSING -eq 0 ]; then
    echo "✅ PASSED: Single source of truth maintained!"
    echo "All documentation is properly organized."
else
    echo "❌ FAILED: Documentation violations detected!"
    echo ""
    echo "Violations found: $VIOLATIONS"
    echo "Missing canonical docs: $MISSING"
    echo ""
    echo "REQUIRED ACTIONS:"
    echo "1. Remove all duplicate/versioned documentation"
    echo "2. Merge content into canonical documents"
    echo "3. Ensure all canonical documents exist"
    echo ""
    echo "Canonical documents (MUST use these):"
    for doc in "${CANONICAL_DOCS[@]}"; do
        echo "   - $doc"
    done
    exit 1
fi
echo "=========================================="

# Optional: Create git hook
if [ "$1" == "--install-hook" ]; then
    echo ""
    echo "Installing pre-commit hook..."
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook to enforce single source of truth

./scripts/enforce_single_source.sh
if [ $? -ne 0 ]; then
    echo "❌ Commit blocked: Fix documentation violations first!"
    exit 1
fi
EOF
    chmod +x .git/hooks/pre-commit
    echo "✅ Pre-commit hook installed!"
fi