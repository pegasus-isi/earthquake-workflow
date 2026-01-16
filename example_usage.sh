#!/bin/bash

# Example usage of earthquake workflow

echo "========================================"
echo "EARTHQUAKE WORKFLOW EXAMPLE USAGE"
echo "========================================"

# Build Docker container (optional, if using locally)
echo ""
echo "1. Build Docker container:"
echo "   cd Docker"
echo "   docker build -f Earthquake_Dockerfile -t kthare10/earthquake-analysis:latest ."
echo "   docker push kthare10/earthquake-analysis:latest"

# Example 1: Single region (California)
echo ""
echo "2. Example 1: California earthquakes (M4.0+, January 2024)"
echo "   ./workflow_generator.py \\"
echo "       --regions california \\"
echo "       --start-date 2024-01-01 \\"
echo "       --end-date 2024-01-31 \\"
echo "       --min-magnitude 4.0 \\"
echo "       --output workflow_california.yml"

# Example 2: Multiple regions
echo ""
echo "3. Example 2: Multiple regions (California, Japan, Indonesia)"
echo "   ./workflow_generator.py \\"
echo "       --regions california japan indonesia \\"
echo "       --start-date 2024-01-01 \\"
echo "       --end-date 2024-01-31 \\"
echo "       --min-magnitude 4.5 \\"
echo "       --output workflow_multi_region.yml"

# Example 3: Pacific Ring of Fire (higher magnitude)
echo ""
echo "4. Example 3: Pacific Ring of Fire (M5.0+)"
echo "   ./workflow_generator.py \\"
echo "       --regions pacific_ring \\"
echo "       --start-date 2024-01-01 \\"
echo "       --end-date 2024-01-31 \\"
echo "       --min-magnitude 5.0 \\"
echo "       --output workflow_pacific.yml"

# Example 4: Worldwide significant earthquakes
echo ""
echo "5. Example 4: Worldwide significant earthquakes (M6.0+)"
echo "   ./workflow_generator.py \\"
echo "       --regions worldwide \\"
echo "       --start-date 2024-01-01 \\"
echo "       --end-date 2024-12-31 \\"
echo "       --min-magnitude 6.0 \\"
echo "       --output workflow_worldwide.yml"

# Submit workflow
echo ""
echo "6. Submit workflow to Pegasus:"
echo "   pegasus-plan --submit -s condorpool -o local workflow.yml"

# Monitor workflow
echo ""
echo "7. Monitor workflow status:"
echo "   pegasus-status /path/to/submit/directory"
echo "   pegasus-analyzer /path/to/submit/directory"

# Check outputs
echo ""
echo "8. Output files will be in:"
echo "   output/output/<region>_catalog.csv"
echo "   output/output/<region>_patterns.json"
echo "   output/output/<region>_visualization.png"
echo "   output/output/<region>_anomalies.json"

echo ""
echo "========================================"
echo "Available regions:"
echo "  - pacific_ring: Pacific Ring of Fire"
echo "  - california: California, USA"
echo "  - japan: Japanese archipelago"
echo "  - indonesia: Indonesian archipelago"
echo "  - turkey: Turkey and surroundings"
echo "  - chile: Chile"
echo "  - worldwide: Global (no bounding box)"
echo "========================================"
