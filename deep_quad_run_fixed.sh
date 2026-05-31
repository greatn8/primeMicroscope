#!/usr/bin/env bash
set -u

CHUNKS="${1:-40}"

BASE_DIR="$(pwd)"
EXE="$BASE_DIR/prime2"

START_BASE=1000000000000
RANGE=200000000
ITERATIONS=1
WIDTH=256
HEIGHT=256

STAMP=$(date +"%Y%m%d_%H%M%S")
OUTDIR="$BASE_DIR/deep_quad_results_${STAMP}"

mkdir -p "$OUTDIR"

echo "Deep run started: $(date)" | tee "$OUTDIR/run_log.txt"
echo "Base dir: $BASE_DIR" | tee -a "$OUTDIR/run_log.txt"
echo "Executable: $EXE" | tee -a "$OUTDIR/run_log.txt"
echo "Chunks: $CHUNKS" | tee -a "$OUTDIR/run_log.txt"
echo "Range per chunk: $RANGE" | tee -a "$OUTDIR/run_log.txt"
echo "Image size: ${WIDTH}x${HEIGHT}" | tee -a "$OUTDIR/run_log.txt"
echo

if [ ! -x "$EXE" ]; then
    echo "ERROR: executable not found or not executable: $EXE" | tee -a "$OUTDIR/run_log.txt"
    echo "Run: module load cuda/auto && make" | tee -a "$OUTDIR/run_log.txt"
    exit 1
fi

MASTER_PATTERNS="$OUTDIR/master_top_patterns.csv"
MASTER_TILES="$OUTDIR/master_top_tiles.csv"
MASTER_FAMILY="$OUTDIR/master_family_summary.csv"

echo "chunk,rank,pattern,max_z,top_count,top_tile,mean,stddev,top_range_start,top_range_end" > "$MASTER_PATTERNS"
echo "chunk,rank,range_start,range_end,novelty,winner_pattern,winner_z" > "$MASTER_TILES"
echo "chunk,family,count,percentage" > "$MASTER_FAMILY"

for ((i=0; i<CHUNKS; i++)); do
    START=$((START_BASE + i * RANGE))
    CHUNK_DIR="$OUTDIR/chunk_$(printf "%03d" "$i")"
    mkdir -p "$CHUNK_DIR"

    echo "============================================================" | tee -a "$OUTDIR/run_log.txt"
    echo "Chunk $i / $((CHUNKS - 1))" | tee -a "$OUTDIR/run_log.txt"
    echo "Start: $START" | tee -a "$OUTDIR/run_log.txt"
    echo "Time: $(date)" | tee -a "$OUTDIR/run_log.txt"
    echo "============================================================" | tee -a "$OUTDIR/run_log.txt"

    (
        cd "$CHUNK_DIR"
        /usr/bin/time -v "$EXE" "$START" "$RANGE" "$ITERATIONS" "$WIDTH" "$HEIGHT" \
            > run_stdout.txt 2> run_time.txt
    )

    STATUS=$?

    if [ "$STATUS" -ne 0 ]; then
        echo "ERROR: chunk $i failed with status $STATUS" | tee -a "$OUTDIR/run_log.txt"
        echo "---- run_stdout.txt ----" | tee -a "$OUTDIR/run_log.txt"
        tail -80 "$CHUNK_DIR/run_stdout.txt" | tee -a "$OUTDIR/run_log.txt"
        echo "---- run_time.txt ----" | tee -a "$OUTDIR/run_log.txt"
        tail -80 "$CHUNK_DIR/run_time.txt" | tee -a "$OUTDIR/run_log.txt"
        exit 1
    fi

    if [ ! -f "$CHUNK_DIR/iter_000_top_patterns.csv" ]; then
        echo "ERROR: chunk $i finished but iter_000_top_patterns.csv was not created" | tee -a "$OUTDIR/run_log.txt"
        echo "Files in chunk:" | tee -a "$OUTDIR/run_log.txt"
        ls -lh "$CHUNK_DIR" | tee -a "$OUTDIR/run_log.txt"
        echo "---- run_stdout.txt ----" | tee -a "$OUTDIR/run_log.txt"
        tail -80 "$CHUNK_DIR/run_stdout.txt" | tee -a "$OUTDIR/run_log.txt"
        echo "---- run_time.txt ----" | tee -a "$OUTDIR/run_log.txt"
        tail -80 "$CHUNK_DIR/run_time.txt" | tee -a "$OUTDIR/run_log.txt"
        exit 1
    fi

    awk -v c="$i" -F, 'NR>1 {print c "," $0}' "$CHUNK_DIR/iter_000_top_patterns.csv" >> "$MASTER_PATTERNS"
    awk -v c="$i" -F, 'NR>1 {print c "," $0}' "$CHUNK_DIR/iter_000_top_tiles.csv" >> "$MASTER_TILES"
    awk -v c="$i" -F, 'NR>1 {print c "," $0}' "$CHUNK_DIR/iter_000_winner_family_summary.csv" >> "$MASTER_FAMILY"

    BMP_COUNT=$(find "$CHUNK_DIR" -name "*.bmp" | wc -l)
    CSV_COUNT=$(find "$CHUNK_DIR" -name "*.csv" | wc -l)

    echo "Chunk $i complete. BMP files: $BMP_COUNT, CSV files: $CSV_COUNT" | tee -a "$OUTDIR/run_log.txt"
    echo "Top patterns for chunk $i:" | tee -a "$OUTDIR/run_log.txt"
    head -8 "$CHUNK_DIR/iter_000_top_patterns.csv" | tee -a "$OUTDIR/run_log.txt"

    echo "GPU status after chunk $i:" | tee -a "$OUTDIR/run_log.txt"
    /usr/bin/nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv | tee -a "$OUTDIR/run_log.txt" || true

    echo | tee -a "$OUTDIR/run_log.txt"
done

echo "============================================================" | tee -a "$OUTDIR/run_log.txt"
echo "DEEP RUN COMPLETE" | tee -a "$OUTDIR/run_log.txt"
echo "Finished: $(date)" | tee -a "$OUTDIR/run_log.txt"
echo "Results: $OUTDIR" | tee -a "$OUTDIR/run_log.txt"
echo "============================================================" | tee -a "$OUTDIR/run_log.txt"

awk -F, 'NR>1 {print $0}' "$MASTER_PATTERNS" | sort -t, -k5,5nr -k4,4nr | head -100 > "$OUTDIR/best_by_top_count.csv"
awk -F, 'NR>1 {print $0}' "$MASTER_PATTERNS" | sort -t, -k4,4nr | head -100 > "$OUTDIR/best_by_max_z.csv"
awk -F, 'NR>1 {print $0}' "$MASTER_TILES" | sort -t, -k5,5nr | head -100 > "$OUTDIR/best_tiles.csv"

echo
echo "Done. Results folder:"
echo "$OUTDIR"
echo
echo "Useful files:"
echo "$OUTDIR/master_top_patterns.csv"
echo "$OUTDIR/master_top_tiles.csv"
echo "$OUTDIR/master_family_summary.csv"
echo "$OUTDIR/best_by_top_count.csv"
echo "$OUTDIR/best_by_max_z.csv"
echo "$OUTDIR/best_tiles.csv"
echo "$OUTDIR/run_log.txt"
