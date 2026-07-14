#!/bin/bash
# Sample the entire ternary surface with 0.05 increments

step=0.05

for a in $(seq 0.05 $step 1); do
  for b in $(seq 0.05 $step 1); do
    fixed=$(echo "1.0 - $a - $b" | bc -l)
    if (( $(echo "$fixed == 0" | bc -l) )); then
      continue
    fi
    if [[ $fixed == .* ]]; then
      fixed="0$fixed"
    fi
    sum=$(awk -v x="$a" -v y="$b" 'BEGIN {print x + y}')
    if (( $(echo "$sum > 1" | bc -l) )); then
      continue
    fi

    dirname="l-Fe-${a}-Cu-${b}-Ni-${fixed}"
    if [[ -d "$dirname" ]]; then
    	echo "Directory exists, skipping..."
    	continue
    fi

    echo "$dirname"
    mkdir -p "$dirname"
    cd "$dirname" || exit 1


    sed -e "s/FRACA/${a}/g" \
        -e "s/FRACB/${b}/g" \
        -e "s/FRACC/${fixed}/g" \
        ../input.lmp > input.lmp


    cd ..
  done
done

