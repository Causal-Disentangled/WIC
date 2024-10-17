for name in 'hailfinder' #'Alarm' 'andes' 'barley' 'Child' 'hailfinder' 'insurance' 'win95pts'
do
    for lambda_1 in 0.01 0.02 0.05 0.1
    do
        for lambda_CI in 0.0 0.25 0.5 1.0
        do
            python3 getResult.py --dataset=$name --lambda_1=$lambda_1 --lambda_3=$lambda_CI
        done
    done
done


