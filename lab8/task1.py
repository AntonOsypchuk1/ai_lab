import numpy as np
import tensorflow as tf

n_samples, batch_size, num_steps = 1000, 100, 20000
X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1)).astype(np.float32)

X_data = (X_data - np.mean(X_data)) / np.std(X_data)
y_data = (y_data - np.mean(y_data)) / np.std(y_data)

k = tf.Variable(tf.random.normal((1, 1), stddev=0.1), name='slope')
b = tf.Variable(tf.zeros((1,)), name='bias')

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

display_step = 100
for i in range(num_steps):
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    with tf.GradientTape() as tape:
        y_pred = tf.matmul(X_batch, k) + b
        loss = tf.reduce_sum((y_batch - y_pred) ** 2)
    
    gradients = tape.gradient(loss, [k, b])
    clipped_gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
    optimizer.apply_gradients(zip(clipped_gradients, [k, b]))

    if (i + 1) % display_step == 0:
        print(f'Epoch {i + 1}: Loss = {loss.numpy():.8f}, k = {k.numpy()[0][0]:.4f}, b = {b.numpy()[0]:.4f}')
