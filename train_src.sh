LOG_PREFIX="pretraining"

DATASETS="cifar100" # cifar10 or cifar100
METHODS="Src"

echo DATASETS: $DATASETS
echo METHODS: $METHODS

GPUS=(0) #available gpus
NUM_GPUS=${#GPUS[@]}

sleep 10s # prevent mistake
if [ ! -d "raw_logs" ]; then
  mkdir raw_logs
fi
# mkdir raw_logs # save console outputs here

#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=3 #num concurrent jobs
  local num_max_jobs=${1:-$default_num_jobs}
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

###############################################################
##### Source Training; Source Evaluation: Source domains  #####
###############################################################
train_source_model() {
  i=0
  update_every_x="64"
  memory_size="64"
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do

      if [ "${DATASET}" = "cifar10" ]; then
        EPOCH=200
        MODEL="resnet18"
      elif [ "${DATASET}" = "cifar100" ]; then
        EPOCH=200
        MODEL="resnet18"
      fi

      for SEED in 0; do
        if [[ "$METHOD" == *"Src"* ]]; then
          ### Train with BN
          python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method Src --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --memory_size ${memory_size} --seed $SEED \
            --log_prefix ${LOG_PREFIX}_${SEED} \
            2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

          i=$((i + 1))
          wait_n

          ## Train with IABN, no fuse
          for iabn_k in 4; do
            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method Src \
              --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} \
              --memory_size ${memory_size} --seed $SEED \
              --iabn --iabn_k ${iabn_k} \
              --log_prefix ${LOG_PREFIX}_${SEED}_iabn_k${iabn_k} \
              2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

            i=$((i + 1))
            wait_n
          done
        fi
      done
    done
  done

  wait
}

train_source_model
