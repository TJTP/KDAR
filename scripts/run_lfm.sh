GPU_ID=0
DEVICE=cuda:${GPU_ID}

DATA_DIR=./data/
DATASET=last-fm

TAU=1
CG_DECAY=0.5
CL_DECAY=8e-5
L2_DECAY=1e-5
ATTN_DR=0.5

LOG_DIR=./kdar_logs/
mkdir -p ${LOG_DIR}

python3 src/main.py \
        --data_dir ${DATA_DIR} \
        --dataset ${DATASET} \
        --device ${DEVICE} \
        --l2_decay ${L2_DECAY} \
        --tau ${TAU} \
        --cl_decay ${CL_DECAY} \
        --cg_decay ${CG_DECAY} \
        --attn_dr ${ATTN_DR} \
        2>&1 | tee -a ${LOG_DIR}/kdar_${DATASET}.log