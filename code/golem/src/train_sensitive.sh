for name in 'Alarm'
do
    for lambda_1 in 0.05 0.1
    do
        for lambda_CI in 0.25 0.5
        do
            for ((seed=1;seed<=5;seed+=1))
            do
                python3 main.py --seed=$seed --n=1000 --dataset=$name --lambda_1=$lambda_1 --lambda_2=5.0 --lambda_3=$lambda_CI --non_equal_variances --checkpoint_iter=5000 --num_iter=40000 --init --init_path='output_sensitive_multi(5)'/$name/equal_var.seed:$seed/lambda_1:$lambda_1.lambda_3:$lambda_CI/B_est.npy
            done
        done
    done
done