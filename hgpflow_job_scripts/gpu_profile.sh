#!/bin/bash
# Background GPU sampler + summary.
#
# Usage:
#   ./gpu_profile.sh <output_csv> [sample_interval_seconds]
#
# Starts sampling nvidia-smi in the background and prints the sampler PID.
# Run your training, then stop with:  kill <PID>
# A summary is printed automatically when the sampler is killed.
#
# Columns: timestamp, gpu_util_%, mem_util_%, mem_used_MiB, power_W, sm_clock_MHz

CSV="${1:-gpu_util.csv}"
INTERVAL="${2:-0.5}"

echo "timestamp,gpu_util,mem_util,mem_used_mib,power_w,sm_clock_mhz" > "$CSV"

# Summary handler: runs when this script receives SIGTERM/SIGINT.
summarize() {
    echo ""
    echo "=== GPU utilization summary ($CSV) ==="
    # skip header, compute stats on gpu_util (col 2) and mem_used (col 4)
    awk -F',' 'NR>1 {
        n++; u=$2; m=$4;
        util[n]=u; sum+=u; if(u>max)max=u;
        msum+=m; if(m>mmax)mmax=m;
    }
    END {
        if (n==0) { print "no samples collected"; exit }
        # sort util for percentiles
        for (i=1;i<=n;i++) for (j=i+1;j<=n;j++) if (util[j]<util[i]) {t=util[i];util[i]=util[j];util[j]=t}
        p50=util[int(n*0.50)+ (n*0.50==int(n*0.50)?0:1)];
        p95=util[int(n*0.95)+ (n*0.95==int(n*0.95)?0:1)];
        printf "samples          : %d\n", n;
        printf "GPU util mean    : %.1f %%\n", sum/n;
        printf "GPU util median  : %.0f %%\n", p50;
        printf "GPU util p95     : %.0f %%\n", p95;
        printf "GPU util max     : %.0f %%\n", max;
        printf "mem used mean    : %.0f MiB\n", msum/n;
        printf "mem used max     : %.0f MiB\n", mmax;
    }' "$CSV"
    echo "======================================="
    exit 0
}
trap summarize TERM INT

echo "Sampling GPU every ${INTERVAL}s -> $CSV"
echo "Sampler PID: $$"
echo "Run your training now. Stop + summarize with:  kill $$"

while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm \
        --format=csv,noheader,nounits >> "$CSV"
    sleep "$INTERVAL"
done
