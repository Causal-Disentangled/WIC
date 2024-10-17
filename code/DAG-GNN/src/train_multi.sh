for name in 'Alarm' # 'Alarm' 'Child' 'hailfinder' 'win95pts' # 'andes' 
do
    for sample_size in 200 500 1000
    do
        for ((seed=1;seed<=6;seed+=1))
        do
            for lambda_CI in 0.25 0.5 1.0
            do
                python3 train_CI_multi.py --CI_order=1 --seed=$seed --data_sample_size=$sample_size --dataset=$name --lambda_CI=$lambda_CI --graph_sem_type=linear-uniform --hard_constraint=1 --significance_level_1=0.05
            done
        done
    done
done