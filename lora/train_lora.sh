export SAMPLE_DIR="/notebooks/lora/test/"
export OUTPUT_DIR="lora/lora_ckpt/log/"

export MODEL_NAME="/datasets/stable-diffusion-diffusers/stable-diffusion-v1-5/"
export LORA_RANK=16

accelerate launch lora/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$SAMPLE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a retrofuturistic comic book artwork of a man firing a laser gun at a large robot" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=2e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --lora_rank=$LORA_RANK \
  --seed="0"   
