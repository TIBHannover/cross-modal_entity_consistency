BASEDIR=$(dirname $0)

for cfg in ${BASEDIR}/../test_yml/news400_*.yml; do
    echo $cfg
    python eval_benchmark.py -c $cfg -o ${BASEDIR}/../resources/results

done