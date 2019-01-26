import tensorflow as tf
import numpy as np
from hparams import hparams as hp

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def vae_weight(global_step):
    warm_up_step = hp.vae_warming_up
    w1 = tf.cond(
       global_step < warm_up_step,
       lambda: tf.cond(
            global_step % 100 < 1,
            lambda: tf.convert_to_tensor(hp.init_vae_weights) + tf.cast(global_step / 100  * hp.vae_weight_multiler, tf.float32),
            lambda: tf.cast(tf.convert_to_tensor(0), tf.float32)
         ),
       lambda: tf.cast(tf.convert_to_tensor(0), tf.float32)
    )
      
    w2 = tf.cond(
       global_step > warm_up_step,
       lambda: tf.cond(
             global_step % 400 < 1,
             lambda: tf.convert_to_tensor(hp.init_vae_weights) + tf.cast((global_step - warm_up_step) / 400 * hp.vae_weight_multiler + warm_up_step / 100 * hp.vae_weight_multiler, tf.float32),
             lambda: tf.cast(tf.convert_to_tensor(0), tf.float32)
         ),
       lambda: tf.cast(tf.convert_to_tensor(0), tf.float32)
    )             
    return tf.maximum(w1, w2)


