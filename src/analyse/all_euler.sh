for start in {0..750..250};
do
  end=$((start+250))
  echo "${start} ${end}"
  START=$start END=$end bash run.euler
done
