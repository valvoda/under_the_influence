for hidden in 50 100 200 300
do
  for dropout in 0.2 0.1 0.3 0.4
  do
    for lr in 0.00003 0.0003 0.000003
    do
      for inp in facts
      do
        for model in longformer
        do
          for arch in joint
          do
            echo ${lr} ${hidden} ${dropout} ${inp} ${model} ${arch}
            LR=$lr HIDDEN=$hidden DROP=$dropout INP=$inp MOD=$model ARCH=$arch sbatch run.wilkes3
          done
        done
      done
    done
  done
done
