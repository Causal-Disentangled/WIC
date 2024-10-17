for ((seed=1;seed<=5;seed+=1))
do
    for name in 'win95pts'
    do
        for lambda_1 in 0.01 0.02 0.05 0.1 0.15 0.2
        do
            for lambda_CI in 0.0
            do
                python3 getResult_multi.py --seed=$seed --dataset=$name --lambda_1=$lambda_1 --lambda_3=$lambda_CI
            done
        done
    done
done