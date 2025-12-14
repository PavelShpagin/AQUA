#!/bin/bash

# Download Missing GEC Datasets
# Downloads Troy-1BW, Troy-Blogs, UberText, and other datasets for GEC training

set -e  # Exit on any error

# Define directories
RAW_DIR="data/raw"
THIRD_PARTY_DIR="third-party"

echo "Downloading Missing GEC Training Datasets"
echo "============================================"

# Create directories
mkdir -p "$RAW_DIR"
mkdir -p "$RAW_DIR/troy-1bw"
mkdir -p "$RAW_DIR/troy-blogs" 
mkdir -p "$RAW_DIR/ubertext"

# Check if cLang-8 exists
echo "1. Checking cLang-8 availability..."
if [ -d "$THIRD_PARTY_DIR/clang8" ] && [ -f "$THIRD_PARTY_DIR/clang8/output_data/clang8_source_target_en.tsv" ]; then
    echo "   cLang-8 already processed and available"
else
    echo "    cLang-8 not found in third-party/clang8/"
    echo "   Please ensure cLang-8 is properly processed in third-party/clang8/"
fi

# Download One Billion Word Benchmark (for Troy-1BW generation)
echo ""
echo "2. Downloading One Billion Word Benchmark (base for Troy-1BW)..."
if [ ! -f "$RAW_DIR/troy-1bw/1-billion-word-language-modeling-benchmark.tar.gz" ]; then
    echo "   Downloading from Google Research..."
    cd "$RAW_DIR/troy-1bw"
    
    # Try to download One Billion Word Benchmark
    curl -L -o "1-billion-word-language-modeling-benchmark.tar.gz" \
        "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz" || \
    echo "   Could not download One Billion Word Benchmark directly"
    
    # Alternative: Try from Google
    if [ ! -f "1-billion-word-language-modeling-benchmark.tar.gz" ]; then
        curl -L -o "1-billion-word-language-modeling-benchmark.tar.gz" \
            "https://drive.google.com/uc?export=download&id=1GIxeMpFAYNElDqMlhsb6-rvFWRQ6Ykhu" || \
        echo "   Google Drive download failed. Manual download required."
    fi
    
    # Extract if download succeeded
    if [ -f "1-billion-word-language-modeling-benchmark.tar.gz" ]; then
        echo "   Extracting One Billion Word Benchmark..."
        tar -xzf "1-billion-word-language-modeling-benchmark.tar.gz"
        echo "   One Billion Word Benchmark downloaded and extracted"
    else
        echo "    Failed to download One Billion Word Benchmark"
        echo "   Manual download required from: http://www.statmt.org/lm-benchmark/"
    fi
    cd - > /dev/null
else
    echo "   One Billion Word Benchmark already exists"
fi

# Download Blog Authorship Corpus (base for Troy-Blogs generation)
echo ""
echo "3. Downloading Blog Authorship Corpus (base for Troy-Blogs)..."
if [ ! -f "$RAW_DIR/troy-blogs/blogs.tar.gz" ]; then
    echo "   Downloading from UCI Repository..."
    cd "$RAW_DIR/troy-blogs"
    
    # Download Blog Authorship Corpus
    curl -L -o "blogs.tar.gz" \
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00304/blogs.tar.gz" || \
    echo "   Could not download Blog Authorship Corpus"
    
    # Extract if download succeeded
    if [ -f "blogs.tar.gz" ]; then
        echo "   Extracting Blog Authorship Corpus..."
        tar -xzf "blogs.tar.gz"
        echo "   Blog Authorship Corpus downloaded and extracted"
    else
        echo "    Failed to download Blog Authorship Corpus"
        echo "   Manual download required from: https://archive.ics.uci.edu/ml/datasets/BlogFeedback"
    fi
    cd - > /dev/null
else
    echo "   Blog Authorship Corpus already exists"
fi

# Download UberText (Ukrainian social media corpus)
echo ""
echo "4. Searching for UberText Ukrainian corpus..."
if [ ! -f "$RAW_DIR/ubertext/ubertext.tar.gz" ]; then
    echo "   Attempting to download UberText..."
    cd "$RAW_DIR/ubertext"
    
    # Try various potential sources for UberText
    # Note: These URLs are speculative as UberText doesn't have a clear public source
    
    echo "   Trying GitHub repositories for UberText..."
    curl -L -o "ubertext.tar.gz" \
        "https://github.com/lang-uk/ubertext/archive/main.tar.gz" 2>/dev/null || \
    echo "   GitHub download failed"
    
    if [ ! -f "ubertext.tar.gz" ]; then
        echo "   Trying alternative sources..."
        curl -L -o "ubertext.zip" \
            "https://huggingface.co/datasets/ubertext/resolve/main/data.zip" 2>/dev/null || \
        echo "   HuggingFace download failed"
    fi
    
    # Check if any download succeeded
    if [ -f "ubertext.tar.gz" ] || [ -f "ubertext.zip" ]; then
        echo "   Extracting UberText..."
        if [ -f "ubertext.tar.gz" ]; then
            tar -xzf "ubertext.tar.gz"
        elif [ -f "ubertext.zip" ]; then
            unzip -q "ubertext.zip"
        fi
        echo "   UberText downloaded and extracted"
    else
        echo "    UberText not found in public repositories"
        echo "   Manual search required - check:"
        echo "     - https://github.com/lang-uk/"
        echo "     - Ukrainian language processing repositories"
        echo "     - Contact authors of MultiGEC 2025 paper"
    fi
    cd - > /dev/null
else
    echo "   UberText already exists"
fi

# Download Troy synthetic datasets (if publicly available)
echo ""
echo "5. Searching for Troy synthetic datasets..."
mkdir -p "$RAW_DIR/troy-synthetic"
cd "$RAW_DIR/troy-synthetic"

echo "   Searching for Troy-1BW synthetic data..."
# Try to find Troy synthetic datasets from paper supplementary materials
curl -L -o "troy-1bw.tar.gz" \
    "https://arxiv.org/src/2203.13064v1/anc/troy-1bw.tar.gz" 2>/dev/null || \
echo "   Troy-1BW not found in arXiv supplementary materials"

curl -L -o "troy-blogs.tar.gz" \
    "https://arxiv.org/src/2203.13064v1/anc/troy-blogs.tar.gz" 2>/dev/null || \
echo "   Troy-Blogs not found in arXiv supplementary materials"

# Try alternative sources
if [ ! -f "troy-1bw.tar.gz" ] && [ ! -f "troy-blogs.tar.gz" ]; then
    echo "   Checking GitHub repositories for Troy datasets..."
    
    # Search for potential repositories
    curl -s "https://api.github.com/search/repositories?q=troy+gec+grammatical+error+correction" | \
    grep -o '"clone_url": "[^"]*' | cut -d'"' -f4 | head -5 > potential_repos.txt
    
    if [ -s potential_repos.txt ]; then
        echo "   Found potential repositories:"
        cat potential_repos.txt | while read repo; do
            echo "     - $repo"
        done
        echo "   Manual inspection required for Troy datasets"
    else
        echo "   No obvious repositories found for Troy datasets"
    fi
    rm -f potential_repos.txt
fi

cd - > /dev/null

# Summary
echo ""
echo "Download Summary"
echo "==================="

# Check what we have
datasets_found=0
datasets_total=4

echo -n "cLang-8: "
if [ -d "$THIRD_PARTY_DIR/clang8" ] && [ -f "$THIRD_PARTY_DIR/clang8/output_data/clang8_source_target_en.tsv" ]; then
    echo "Available (2.37M EN, 114K DE, 44K RU entries)"
    ((datasets_found++))
else
    echo " Missing"
fi

echo -n "One Billion Word Benchmark: "
if [ -f "$RAW_DIR/troy-1bw/1-billion-word-language-modeling-benchmark.tar.gz" ] || [ -d "$RAW_DIR/troy-1bw/1-billion-word-language-modeling-benchmark-r13output" ]; then
    echo "Available (base for Troy-1BW generation)"
    ((datasets_found++))
else
    echo " Missing"
fi

echo -n "Blog Authorship Corpus: "
if [ -f "$RAW_DIR/troy-blogs/blogs.tar.gz" ] || [ -d "$RAW_DIR/troy-blogs/blogs" ]; then
    echo "Available (base for Troy-Blogs generation)"
    ((datasets_found++))
else
    echo " Missing"
fi

echo -n "UberText: "
if [ -f "$RAW_DIR/ubertext/ubertext.tar.gz" ] || [ -f "$RAW_DIR/ubertext/ubertext.zip" ] || [ -d "$RAW_DIR/ubertext/ubertext-main" ]; then
    echo "Available"
    ((datasets_found++))
else
    echo " Missing"
fi

echo ""
echo "Datasets available: $datasets_found/$datasets_total"

if [ $datasets_found -lt $datasets_total ]; then
    echo ""
    echo "Manual Download Instructions"
    echo "================================"
    echo "Some datasets require manual download due to:"
    echo "  - Academic access requirements"
    echo "  - Restricted distribution"
    echo "  - Private repositories"
    echo ""
    echo "Please refer to the following papers for dataset access:"
    echo "  - Pillars of GEC: https://arxiv.org/pdf/2404.14914"
    echo "  - Troy synthetic data: https://arxiv.org/abs/2203.13064"
    echo "  - MultiGEC 2025: https://aclanthology.org/2025.unlp-1.17.pdf"
fi

echo ""
echo "🎯 Next Steps"
echo "============="
echo "1. Run processing scripts to use available datasets:"
echo "   python processing/process_bea_training.py --output data/training/bea --max"
echo "   python processing/process_multigec2025_training.py --output data/training/multigec2025 --max"
echo ""
echo "2. Update processing scripts to include new datasets when available"
echo ""
echo "Download script completed!"
