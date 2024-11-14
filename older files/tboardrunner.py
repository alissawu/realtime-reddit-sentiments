#tensorboard --logdir="C:\Users\epw268\Documents\GitHub\realtime-reddit-sentiments\older files\logs\validation"

import tensorflow as tf

for event in tf.compat.v1.train.summary_iterator("C:\Users\epw268\Documents\GitHub\realtime-reddit-sentiments\older files\logs\validation"):
    for value in event.summary.value:
        print(f"Step: {event.step}, Tag: {value.tag}, Value: {value.simple_value}")


