
SRC_PREFIX="pretraining"
LOG_PREFIX="methods"

DATASETS=("cifar10") # cifar10 or cifar100
METHODS=("TENT" "CoTTA" "LAME") #Src BN_Stats ONDA PseudoLabel TENT CoTTA NOTE NOTE_iid

echo DATASETS: ${DATASETS[@]}
echo METHODS: ${METHODS[@]}

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


test_time_adaptation() {
  ###############################################################
  ###### Run Baselines & NOTE; Evaluation: Target domains  ######
  ###############################################################

  i=0

  for DATASET in ${DATASETS[@]}; do
    for METHOD in ${METHODS[@]}; do

      update_every_x="64"
      memory_size="64"
      SEED="0"
      lr="0.001" #other baselines
      weight_decay="0"
      if [ "${DATASET}" = "cifar10" ]; then
        MODEL="resnet18"
        CP_base="log/cifar10/Src/"${SRC_PREFIX} # pretrained models on source domain

      elif [ "${DATASET}" = "cifar100" ]; then
        MODEL="resnet18"
        CP_base="log/cifar100/Src/"${SRC_PREFIX} # pretrained models on source domain
      fi

      for SEED in 0 1 2; do #multiple seeds
          if [ "${METHOD}" = "Src" ]; then
            EPOCH=0
            #### Direct evaluation of source model
            CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET \
              --method ${METHOD} --model $MODEL --epoch $EPOCH \
              -update_every_x ${update_every_x} --load_checkpoint_path ${CP} --seed $SEED \
              --log_prefix ${LOG_PREFIX}_${SEED} \
              2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

            i=$((i + 1))
            wait_n

        elif [ "${METHOD}" = "NOTE" ]; then

          lr="0.0001"
          EPOCH=1
          memory_type="PBRS"
          loss_scaler=0
          iabn_k=4
          bn_momentum=0.01
          #### Train with IABN
          CP=${CP_base}_${SEED}_iabn_k${iabn_k}/cp/cp_last.pth.tar
          python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} \
            --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
            --remove_cp --online --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} \
            --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} \
            --bn_momentum ${bn_momentum} \
            --iabn --iabn_k ${iabn_k} \
            --log_prefix ${LOG_PREFIX}_${SEED}_iabn_k${iabn_k}_mt${bn_momentum} \
            --loss_scaler ${loss_scaler} \
            ${validation} \
            2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

          i=$((i + 1))
          wait_n

        elif [ "${METHOD}" = "BN_Stats" ]; then
            EPOCH=1
            #### Train with BN
            CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} \
            --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
            --remove_cp --online --update_every_x ${update_every_x} \
            --memory_size ${memory_size} \
            --log_prefix ${LOG_PREFIX}_${SEED} \
            ${validation} \
            2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

            i=$((i + 1))
            wait_n

        elif [ "${METHOD}" = "ONDA" ]; then

          EPOCH=1
          #### Train with BN
          update_every_x=10
          memory_size=10
          bn_momentum=0.1
          CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
          python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} \
            --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
            --remove_cp --online --use_learned_stats --weight_decay ${weight_decay} \
            --update_every_x ${update_every_x} --memory_size ${memory_size} \
            --bn_momentum ${bn_momentum} \
            --log_prefix ${LOG_PREFIX}_${SEED} \
            ${validation} \
            2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

          i=$((i + 1))
          wait_n

        elif [ "${METHOD}" = "PseudoLabel" ]; then
          EPOCH=1
          lr=0.001
          #### Train with BN
          CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
          python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET \
            --method ${METHOD} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
            --remove_cp --online --update_every_x ${update_every_x} \
            --memory_size ${memory_size} \
            --lr ${lr} --weight_decay ${weight_decay} \
            --log_prefix ${LOG_PREFIX}_${SEED} \
            2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

          i=$((i + 1))
          wait_n

        elif [ "${METHOD}" = "TENT" ]; then
          EPOCH=1
          lr=0.001
          #### Train with BN
          CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
          python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} \
            --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
            --remove_cp --online --update_every_x ${update_every_x} --memory_size ${memory_size} \
            --lr ${lr} --weight_decay ${weight_decay} \
            --log_prefix ${LOG_PREFIX}_${SEED} \
              2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

          i=$((i + 1))
          wait_n

        elif [ "${METHOD}" = "LAME" ]; then
          EPOCH=1
          #### Train with BN
          CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
          python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} \
            --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
            --remove_cp --online --update_every_x ${update_every_x} --memory_size ${memory_size} \
            --lr ${lr} --weight_decay ${weight_decay} \
            --log_prefix ${LOG_PREFIX}_${SEED} \
              2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

            i=$((i + 1))
            wait_n

        elif [ "${METHOD}" = "CoTTA" ]; then
          lr=0.001
          EPOCH=1

          if [ "${DATASET}" = "cifar10" ]; then
            aug_threshold=0.92 #value reported from the official code
          elif [ "${DATASET}" = "cifar100" ]; then
            aug_threshold=0.72 #value reported from the official code
          fi

          CP=${CP_base}_${SEED}/cp/cp_last.pth.tar
          python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --model $MODEL --epoch $EPOCH --load_checkpoint_path ${CP} --seed $SEED \
            --remove_cp --online --update_every_x ${update_every_x} --memory_size ${memory_size} \
            --lr ${lr} --weight_decay ${weight_decay} \
            --aug_threshold ${aug_threshold} \
            --log_prefix ${LOG_PREFIX}_${SEED} \
              2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

          i=$((i + 1))
          wait_n
        fi

      done
    done
  done

  wait
}

test_time_adaptation