#!/bin/bash

GOALS=(
    "alpha_helices_hydrophilic"
    "beta_strands"
    "loose_coils"
    "alpha_helices_alanine_leucine_glutamate"
)

ITERATIONS=${1:-64}

echo "Running all design goals individually with energy monitoring"
echo "=============================================================="
echo "Iterations per goal: $ITERATIONS"
echo "Total goals: ${#GOALS[@]}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENERGY_MONITOR="../analysis/energy_monitor.py"

cd "$SCRIPT_DIR"

if [[ ! -f "$ENERGY_MONITOR" ]]; then
    echo "❌ Error: Energy monitor not found at $ENERGY_MONITOR"
    exit 1
fi

if [[ "$CONDA_SETUP_DONE" == "1" ]]; then
    echo "✅ Using conda environment from parent process: $CONDA_DEFAULT_ENV"
    echo "   Python: $(which python)"
    echo "   GPU devices: $CUDA_VISIBLE_DEVICES"
    
    python -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "   ⚠️  GPU check failed"
    
    python -c "import pyrosetta; print('   PyRosetta: Available')" || {
        echo "❌ PyRosetta not available - this should have been installed by the sbatch script"
        echo "   Attempting emergency installation..."
        pip install pyrosetta-installer
        python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
        python -c "import pyrosetta; print('   PyRosetta: Now available')" || {
            echo "❌ Emergency PyRosetta installation failed"
            exit 1
        }
    }
elif command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate ProteinSwarm
    echo "✅ Conda environment 'ProteinSwarm' activated"
    python -c "import pyrosetta; print('   PyRosetta: Available')" || {
        echo "❌ PyRosetta not available after activation"
        exit 1
    }
else
    echo "❌ Error: conda not found and no parent environment"
    exit 1
fi

BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="experiment_alpha_helices_15_${BATCH_TIMESTAMP}"
mkdir -p "$BATCH_DIR"

echo "📁 Batch results directory: $BATCH_DIR"
echo ""

run_goal_with_monitoring() {
    local goal_name=$1
    local iterations=$2
    local batch_dir=$3
    
    echo "🎯 Starting goal: $goal_name"
    echo "=================================="
    
    local goal_timestamp=$(date +%Y%m%d_%H%M%S)
    local goal_dir="${batch_dir}/${goal_name}_${goal_timestamp}"
    mkdir -p "$goal_dir"
    
    local log_file="${goal_dir}/${goal_name}_${goal_timestamp}.log"
    local csv_file="${goal_dir}/${goal_name}_energy_scores_${goal_timestamp}.csv"
    local pid_file="${goal_dir}/${goal_name}.pid"
    local monitor_pid_file="${goal_dir}/${goal_name}_monitor.pid"
    
    echo "  📄 Log file: $log_file"
    echo "  📊 CSV file: $csv_file"
    echo ""
    
    echo "  🔄 Starting design process..."
    echo "  🐍 Using Python: $(which python)"
    echo "  🧠 Memory before start: $(free -m | awk '/^Mem:/{printf "%.1f GB used", $3/1024}')"
    
    python -c "import gc; gc.collect()" 2>/dev/null
    
    nohup python test_goal_design.py $iterations --goal $goal_name > "$log_file" 2>&1 &
    local design_pid=$!
    echo $design_pid > "$pid_file"
    
    sleep 3
    
    echo "  📈 Starting energy monitoring..."
    nohup python "$ENERGY_MONITOR" "$log_file" -o "$csv_file" --parse-existing > "${goal_dir}/monitor.log" 2>&1 &
    local monitor_pid=$!
    echo $monitor_pid > "$monitor_pid_file"
    
    if ps -p $design_pid > /dev/null && ps -p $monitor_pid > /dev/null; then
        echo "  ✅ Both processes started successfully"
        echo "     Design PID: $design_pid"
        echo "     Monitor PID: $monitor_pid"
        
        echo "  ⏳ Waiting for design to complete..."
        wait $design_pid
        local design_exit_code=$?
        
        sleep 5
        
        if ps -p $monitor_pid > /dev/null; then
            kill $monitor_pid 2>/dev/null
            echo "  🛑 Monitor stopped"
        fi
        
        rm -f "$pid_file" "$monitor_pid_file"
        
        if [[ $design_exit_code -eq 0 ]]; then
            echo "  ✅ Goal $goal_name completed successfully"
        else
            echo "  ❌ Goal $goal_name failed with exit code $design_exit_code"
        fi
        
        if [[ -f "$csv_file" ]]; then
            local csv_lines=$(wc -l < "$csv_file")
            echo "  📊 Energy CSV has $csv_lines entries"
            echo "  📈 Energy progression:"
            tail -3 "$csv_file" | cut -d',' -f1,2,3,6 | sed 's/^/     /'
        fi
        
    else
        echo "  ❌ Failed to start processes for $goal_name"
        [[ -f "$pid_file" ]] && rm -f "$pid_file"
        [[ -f "$monitor_pid_file" ]] && rm -f "$monitor_pid_file"
        return 1
    fi
    
    echo ""
    return $design_exit_code
}

echo "🏃 Starting sequential execution of all goals..."
echo ""

SUCCESS_COUNT=0
TOTAL_COUNT=${#GOALS[@]}
FAILED_GOALS=()

for goal in "${GOALS[@]}"; do
    echo "[$((SUCCESS_COUNT + ${#FAILED_GOALS[@]} + 1))/$TOTAL_COUNT] Processing: $goal"
    
    if run_goal_with_monitoring "$goal" "$ITERATIONS" "$BATCH_DIR"; then
        ((SUCCESS_COUNT++))
        echo "✅ $goal completed successfully"
    else
        FAILED_GOALS+=("$goal")
        echo "❌ $goal failed"
    fi
    
    echo "----------------------------------------"
    echo ""
done

echo "🏁 BATCH COMPLETION SUMMARY"
echo "=========================="
echo "Total goals: $TOTAL_COUNT"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: ${#FAILED_GOALS[@]}"

if [[ ${#FAILED_GOALS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed goals:"
    for failed_goal in "${FAILED_GOALS[@]}"; do
        echo "  ❌ $failed_goal"
    done
fi

echo ""
echo "📁 All results saved in: $BATCH_DIR"
echo "📊 Each goal has its own energy monitoring CSV file"
echo ""
echo "🔍 Quick result check:"
find "$BATCH_DIR" -name "*energy_scores*.csv" | head -5 | while read csv; do
    goal_name=$(basename "$(dirname "$csv")" | cut -d'_' -f1)
    lines=$(wc -l < "$csv" 2>/dev/null || echo "0")
    echo "  📈 $goal_name: $lines entries"
done

if [[ ${#FAILED_GOALS[@]} -eq 0 ]]; then
    echo "🎉 All goals completed successfully!"
    exit 0
else
    echo "⚠️  Some goals failed. Check individual logs for details."
    exit 1
fi 