#!/bin/bash
set -x
grep "ENTRY" sim.log | wc -l
grep "BOOTSTRAP" sim.log | wc -l
grep "FWD origin" sim.log | head -20
grep "BWD origin" sim.log | head -20
grep "FWD rx done" sim.log | wc -l
grep "BWD rx done" sim.log | wc -l
grep "FWD tx done" sim.log | wc -l
grep "BWD tx done" sim.log | wc -l
grep "FINALIZE" sim.log | wc -l
