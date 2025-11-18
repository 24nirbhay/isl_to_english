import tensorflow as tf


def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('No GPU detected by TensorFlow.')
        return False
    print(f'Detected GPUs: {gpus}')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return True


if __name__ == '__main__':
    ok = check_gpu()
    if ok:
        print('TensorFlow can access GPU.\n')
        # Simple test: allocate a small tensor on GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.linalg.matmul(a, a)
            print('Matrix multiply result:\n', b.numpy())
    else:
        print('GPU not available. Running on CPU.')
