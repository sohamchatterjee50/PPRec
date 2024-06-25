python train.py \
    --data-folder ~/Data/RecSys/EBNerd \
    --output ~/Data/RecSys/Runs/first \
    --model-config minimal \
    --train-config paper \
    --max-clicked 20 \
    --max-epochs 10 \
    --data-size mini \
    --device mps \
    --description "Testing whether the model can be trained on the mini dataset" \
    --log


