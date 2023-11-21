for start in {0..900..50};
do
  for model in longformer
  do
    for arch in joint
    do
      end=$((start+50))
      echo "${start} ${end} ${arch} ${model}"
      START=$start END=$end MOD=$model ARCH=$arch sbatch run.wilkes3
    done
  done
done