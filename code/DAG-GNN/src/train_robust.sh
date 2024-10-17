for name in 'Alarm' # 'Alarm' 'Child' 'hailfinder' 'win95pts' # 'andes' 
do
    for sample_size in 1000
    do
        for ((seed=2;seed<=2;seed+=1))
        do
            for lambda_CI in 0.5
            do
                for ((set_errors=3;set_errors<=3;set_errors+=1))
                do
                    python3 train_CI_robust.py --CI_order=0 --seed=$seed --set_errors=$set_errors --data_sample_size=$sample_size --dataset=$name --lambda_CI=$lambda_CI --graph_sem_type=linear-uniform --hard_constraint=1
                done
            done
        done
    done
done