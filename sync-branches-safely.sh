#!/bin/bash
# Safe strategy to sync working-on-the-documentation branch with main
# This preserves the working branch and creates a backup before any changes

echo "🔄 Safe Branch Sync Strategy"
echo "================================"
echo ""

# Step 1: Create a backup of current main
echo "📦 Step 1: Creating backup of main branch..."
git branch main-backup-$(date +%Y%m%d-%H%M%S) main
echo "   ✅ Backup created"
echo ""

# Step 2: Check current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "📍 Current branch: $CURRENT_BRANCH"
echo ""

# Step 3: Switch to working branch if not already there
if [ "$CURRENT_BRANCH" != "working-on-the-documentation" ]; then
    echo "🔀 Step 2: Switching to working-on-the-documentation..."
    git checkout working-on-the-documentation
    echo ""
fi

# Step 4: Push working branch to remote (requires authentication)
echo "⬆️  Step 3: Pushing working-on-the-documentation to remote..."
echo "   (You'll need to authenticate)"
git push origin working-on-the-documentation
if [ $? -eq 0 ]; then
    echo "   ✅ Working branch pushed successfully"
else
    echo "   ⚠️  Push failed - continuing anyway (you can push manually later)"
fi
echo ""

# Step 5: Update main to match working branch
echo "🔄 Step 4: Updating main to match working-on-the-documentation..."
git checkout main
git reset --hard working-on-the-documentation
echo "   ✅ Main now matches working-on-the-documentation"
echo ""

# Step 6: Push main (requires force push since history changed)
echo "⬆️  Step 5: Pushing updated main to remote..."
echo "   ⚠️  This requires --force-with-lease (safer than --force)"
echo "   (You'll need to authenticate)"
git push origin main --force-with-lease
if [ $? -eq 0 ]; then
    echo "   ✅ Main pushed successfully"
else
    echo "   ❌ Push failed - you'll need to push manually:"
    echo "      git push origin main --force-with-lease"
fi
echo ""

# Step 7: Return to working branch
echo "🔀 Step 6: Returning to working-on-the-documentation..."
git checkout working-on-the-documentation
echo ""

echo "✅ Branch sync complete!"
echo ""
echo "Summary:"
echo "  • Backup created (check 'git branch' for main-backup-* branch)"
echo "  • Main now matches working-on-the-documentation"
echo "  • Both branches pushed to remote (if authentication succeeded)"
echo ""
echo "If push failed due to authentication, run manually:"
echo "  git push origin working-on-the-documentation"
echo "  git checkout main"
echo "  git push origin main --force-with-lease"
echo "  git checkout working-on-the-documentation"

