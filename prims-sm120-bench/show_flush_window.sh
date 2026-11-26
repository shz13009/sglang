#!/usr/bin/env bash
# Show the prefill batches right after the Nth flush_cache in a server log.
F=${1:-~/code/prims-sm120-bench/e2e/on/server.log}
K=${2:-3}
N=$(grep -n "flush_cache" "$F" | sed -n "${K}p" | cut -d: -f1)
tail -n +"$N" "$F" | grep -E "Prefill batch|flush_cache" | head -14 | cut -c30-200
