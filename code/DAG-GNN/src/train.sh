for name in 'hailfinder' # 'Child' 'hailfinder' 'win95pts' # 'andes' 
do
    for sample_size in 200 500 1000 2000
    do
        for lambda_CI in 0.25 0.5 1.0
        do
            python3 train_CI.py --data_sample_size=$sample_size --dataset=$name --lambda_CI=$lambda_CI --graph_sem_type=linear-uniform
        done
    done
done