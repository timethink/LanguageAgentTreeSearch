python run.py \
    --backend deepseek-chat \
    --task_start_index 0 \
    --task_end_index 5 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 30 \
    --log logs/tot_10k.log \
    --algorithm lats \
    ${@}
