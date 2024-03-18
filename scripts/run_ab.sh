GPU_ID=0
DEVICE=cuda:${GPU_ID}

DATA_DIR=./data/
DATASET=amazon_book

TAU=1
CG_DECAY=0.01
CL_DECAY=5e-8
L2_DECAY=1e-5
ATTN_DR=0.005

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