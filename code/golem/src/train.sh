for name in 'Alarm' #'barley' 'Child' 'hailfinder' 'win95pts' # 'andes'
do
    for lambda_1 in 0.05 0.1 0.15 0.2
    do
        for lambda_CI in 0.25 0.5 1.0
        do
            for ((seed=1;seed<=5;seed+=1))
            do
                for sz in 200 500 1000
                do
                    python3 main.py --seed=$seed --n=$sz --dataset=$name --lambda_1=$lambda_1 --lambda_2=5.0 --lambda_3=$lambda_CI --equal_variances --checkpoint_iter=5000 --num_iter=40000 #--init --init_path=output_last_$sz/$name/equal_var.seed:$seed/lambda_1:$lambda_1.lambda_3:$lambda_CI/B_est.npy
                done
            done
        done
    done
done
