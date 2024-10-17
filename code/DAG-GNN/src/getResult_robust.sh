for name in 'hailfinder' # 'Alarm' 'andes' 'barley' 'Child' 'hailfinder' 'insurance' 'win95pts'
do
    for ((seed=1;seed<=6;seed+=1))
    do
        for samples in 200 500 1000
        do
            for lambda_CI in 0.5
            do
                for ((errors=0;errors<=3;errors+=1))
                do
                    python3 getResult_robust.py --dataset=$name --seed=$seed --samples=$samples --error=$errors --lambda_CI=$lambda_CI
                done
            done
        done
    done
done


