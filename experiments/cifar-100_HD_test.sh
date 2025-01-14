# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0'
CONFIG=configs/cifar-100_prompt_vq.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR
HDDIM=10000


# VQ-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size, default equal to task number
#    arg 2 = prompt length, default 8
#    arg 3 = temperature

for dec_N in 100 10
do
    python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name HDPrompt \
        --prompt_param 10 8 1 --seeds 0\
        --hd_dim $HDDIM --dec_N $dec_N\
        --log_dir ${OUTDIR}/hd-prompt-HDDec${dec_N}
    sleep 10
done