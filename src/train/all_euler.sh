for hidden in 50 100 200 300
do
  for dropout in 0.1 0.2 0.3 0.4
  do
    for lr in 0.00003 0.0003 0.000003
    do
      for inp in facts arguments
      do
        echo ${lr} ${hidden} ${dropout} ${inp}
        LR=$lr HIDDEN=$hidden DROP=$dropout  INP=$inp bash run.euler
      done
    done
  done
done
