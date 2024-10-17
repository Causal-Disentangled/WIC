for name in 'Alarm' # 'Alarm' 'andes' 'barley' 'Child' 'hailfinder' 'insurance' 'win95pts'
do
    for ((seed=1;seed<=6;seed+=1))
    do
        for samples in 200 500 1000
        do
            for lambda_CI in 0.25 0.5 1.0
            do
                python3 getResult.py --dataset=$name --seed=$seed --samples=$samples --lambda_CI=$lambda_CI
            done
        done
    done
done


