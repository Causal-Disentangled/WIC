for ((seed=1;seed<=5;seed+=1))
do
    for name in 'Alarm'
    do
        for lambda_1 in 0.05 0.1
        do
            for lambda_CI in 0.25 0.5
            do
                python3 getResult_sensitive.py --seed=$seed --dataset=$name --lambda_1=$lambda_1 --lambda_3=$lambda_CI
            done
        done
    done
done