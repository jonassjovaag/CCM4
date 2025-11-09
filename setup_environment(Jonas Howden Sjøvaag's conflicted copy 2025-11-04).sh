#!/bin/bash
# MusicHal 9000 - Environment Setup Script
# 
# This script recreates the CCM3 virtual environment from requirements.txt
# Run with: bash setup_environment.sh

echo "🎵 Setting up MusicHal 9000 environment..."

# Check if Python 3.10 is available
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

echo "✅ Using Python: $($PYTHON_CMD --version)"

# Create virtual environment
if [ -d "CCM3" ]; then
    echo "⚠️  CCM3 directory already exists. Remove it? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf CCM3
    else
        echo "❌ Setup cancelled"
        exit 1
    fi
fi

echo "🔄 Creating virtual environment CCM3..."
$PYTHON_CMD -m venv CCM3

# Activate environment
echo "🔄 Activating environment..."
source CCM3/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "🔄 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use MusicHal 9000:"
echo "1. Activate environment: source CCM3/bin/activate"
echo "2. Train a model: python Chandra_trainer.py --file input_audio/your_file.wav"
echo "3. Run live: python MusicHal_9000.py"
echo ""
echo "🎵 Happy improvising!"