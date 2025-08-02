#!/bin/bash

# Release script for Quant Trading Service
# Usage: ./scripts/release.sh [patch|minor|major]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on main branch
if [[ $(git branch --show-current) != "main" ]]; then
    print_error "Must be on main branch to create a release"
    exit 1
fi

# Check if working directory is clean
if [[ -n $(git status --porcelain) ]]; then
    print_error "Working directory is not clean. Please commit or stash changes."
    exit 1
fi

# Get current version
CURRENT_VERSION=$(cat VERSION)
print_status "Current version: $CURRENT_VERSION"

# Determine version bump type
BUMP_TYPE=${1:-patch}

if [[ ! "$BUMP_TYPE" =~ ^(patch|minor|major)$ ]]; then
    print_error "Invalid bump type. Use patch, minor, or major"
    exit 1
fi

# Calculate new version
IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

case $BUMP_TYPE in
    patch)
        PATCH=$((PATCH + 1))
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
print_status "New version: $NEW_VERSION"

# Update version file
echo "$NEW_VERSION" > VERSION
print_success "Updated VERSION file"

# Update release notes
RELEASE_DATE=$(date +"%B %d, %Y")
TAG_NAME="v$NEW_VERSION"

# Create release notes entry
cat > RELEASE_NOTES.md << EOF
# 🚀 Release Notes - Quant Trading Service $NEW_VERSION

## 📅 Release Date
**$RELEASE_DATE**

## 🎯 Version $NEW_VERSION

### ✨ New Features
<!-- Add new features here -->

### 🔧 Technical Improvements
<!-- Add technical improvements here -->

### 🐛 Bug Fixes
<!-- Add bug fixes here -->

### 📈 Performance Improvements
<!-- Add performance improvements here -->

### 📚 Documentation
<!-- Add documentation updates here -->

---

**Built with ❤️ for quantitative trading enthusiasts**

**For support and questions, please refer to the documentation or create an issue on GitHub.**
EOF

print_success "Updated RELEASE_NOTES.md"

# Commit changes
git add VERSION RELEASE_NOTES.md
git commit -m "🚀 Release $NEW_VERSION

📦 Release Changes:
- Updated version to $NEW_VERSION
- Updated release notes for $NEW_VERSION
- Prepared for release deployment

🎯 Release Type: $BUMP_TYPE
📅 Release Date: $RELEASE_DATE"

print_success "Committed version changes"

# Create and push tag
git tag -a "$TAG_NAME" -m "Release $NEW_VERSION"
git push origin main
git push origin "$TAG_NAME"

print_success "Created and pushed tag $TAG_NAME"

# Create GitHub release
gh release create "$TAG_NAME" \
    --title "Release $NEW_VERSION" \
    --notes-file RELEASE_NOTES.md \
    --latest

print_success "Created GitHub release"

# Print summary
echo ""
print_success "Release $NEW_VERSION created successfully!"
echo ""
echo "📋 Release Summary:"
echo "  Version: $NEW_VERSION"
echo "  Tag: $TAG_NAME"
echo "  Date: $RELEASE_DATE"
echo "  Type: $BUMP_TYPE"
echo ""
echo "🔗 GitHub Release: https://github.com/rajamohan1950/quant_trading_service/releases/tag/$TAG_NAME"
echo ""
echo "🚀 Next Steps:"
echo "  1. Review the release on GitHub"
echo "  2. Deploy to staging environment"
echo "  3. Run integration tests"
echo "  4. Deploy to production"
echo "  5. Monitor application health" 