import os
import time
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configurar memoria dinámica en GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU activada: {gpus[0].name}")
else:
    print("ADVERTENCIA: Ejecutando en CPU.")

# --- 1. RUTAS Y PARÁMETROS ---
BASE_DIR = 'dataset'
PATH_TRAIN_A = os.path.join(BASE_DIR, 'trainA')
PATH_TRAIN_B = os.path.join(BASE_DIR, 'trainB')
PATH_TEST_A  = os.path.join(BASE_DIR, 'testA')
PATH_TEST_B  = os.path.join(BASE_DIR, 'testB')

os.makedirs('resultados', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 15
EPOCHS = 80

# --- 2. PIPELINE DE DATOS ---
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    return tf.cast(image, tf.float32)

def random_jitter(image):
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    image = tf.image.random_flip_left_right(image)
    return image

def normalize(image):
    return (image / 127.5) - 1.0

def preprocess_train(image_file):
    return normalize(random_jitter(load_image(image_file)))

def preprocess_test(image_file):
    image = tf.image.resize(load_image(image_file), [IMG_HEIGHT, IMG_WIDTH])
    return normalize(image)

def crear_dataset(folder_path, is_train=True):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    ds = tf.data.Dataset.from_tensor_slices(files)
    if is_train:
        ds = ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    else:
        ds = ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE), len(files)

train_dia, n_train_a = crear_dataset(PATH_TRAIN_A, is_train=True)
train_noche, n_train_b = crear_dataset(PATH_TRAIN_B, is_train=True)
test_dia, n_test_a = crear_dataset(PATH_TEST_A, is_train=False)
test_noche, n_test_b = crear_dataset(PATH_TEST_B, is_train=False)

print(f"Dataset cargado -> Train: {n_train_a} día / {n_train_b} noche | Test: {n_test_a} día / {n_test_b} noche")

# --- 3. ARQUITECTURA (ResNet Generator + PatchGAN Discriminator) ---
class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02), trainable=True)
        self.offset = self.add_weight(
            name='offset', shape=input_shape[-1:],
            initializer='zeros', trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        return self.scale * ((x - mean) * inv) + self.offset

class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)

    def call(self, input_tensor):
        pw, ph = self.padding
        return tf.pad(input_tensor, [[0, 0], [ph, ph], [pw, pw], [0, 0]], mode='REFLECT')

def residual_block(x, filters, kernel_size=3):
    init = tf.random_normal_initializer(0., 0.02)
    skip = x
    x = ReflectionPadding2D((1, 1))(x)
    x = layers.Conv2D(filters, kernel_size, padding='valid', kernel_initializer=init, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = ReflectionPadding2D((1, 1))(x)
    x = layers.Conv2D(filters, kernel_size, padding='valid', kernel_initializer=init, use_bias=False)(x)
    x = InstanceNormalization()(x)
    return layers.Add()([skip, x])

def resnet_generator(num_res_blocks=6):
    init = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    x = ReflectionPadding2D((3, 3))(inputs)
    x = layers.Conv2D(64, 7, padding='valid', kernel_initializer=init, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    for filters in [128, 256]:
        x = layers.Conv2D(filters, 3, strides=2, padding='same', kernel_initializer=init, use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)

    for _ in range(num_res_blocks):
        x = residual_block(x, 256)

    for filters in [128, 64]:
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = ReflectionPadding2D((1, 1))(x)
        x = layers.Conv2D(filters, 3, padding='valid', kernel_initializer=init, use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = layers.ReLU()(x)

    x = ReflectionPadding2D((3, 3))(x)
    outputs = layers.Conv2D(3, 7, padding='valid', kernel_initializer=init, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def downsample_disc(filters, size, apply_norm=True):
    init = tf.random_normal_initializer(0., 0.02)
    res = tf.keras.Sequential()
    res.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=init, use_bias=False))
    if apply_norm:
        res.add(InstanceNormalization())
    res.add(layers.LeakyReLU())
    return res

def discriminator():
    init = tf.random_normal_initializer(0., 0.02)
    inp = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    d1 = downsample_disc(64, 4, False)(inp)
    d2 = downsample_disc(128, 4)(d1)
    d3 = downsample_disc(256, 4)(d2)
    pad1 = layers.ZeroPadding2D()(d3)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=init, use_bias=False)(pad1)
    norm = InstanceNormalization()(conv)
    act = layers.LeakyReLU()(norm)
    pad2 = layers.ZeroPadding2D()(act)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=init)(pad2)
    return tf.keras.Model(inputs=inp, outputs=last)

generator_g = resnet_generator(6)  # Día -> Noche
generator_f = resnet_generator(6)  # Noche -> Día
discriminator_x = discriminator()
discriminator_y = discriminator()

# --- 4. PÉRDIDAS Y OPTIMIZADORES ---
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    return (loss_obj(tf.ones_like(real), real) + loss_obj(tf.zeros_like(generated), generated)) * 0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))

def identity_loss(real_image, same_image):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))

gen_g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
gen_f_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_x_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_y_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

ckpt = tf.train.Checkpoint(
    generator_g=generator_g, generator_f=generator_f,
    discriminator_x=discriminator_x, discriminator_y=discriminator_y,
    gen_g_opt=gen_g_opt, gen_f_opt=gen_f_opt,
    disc_x_opt=disc_x_opt, disc_y_opt=disc_y_opt
)
ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f"Checkpoint previo restaurado: {ckpt_manager.latest_checkpoint}")

def guardar_muestra(model, test_input, epoch):
    prediction = model(test_input, training=False)
    fig = plt.figure(figsize=(10, 5))
    imgs = [test_input[0], prediction[0]]
    titulos = ['Real (Día)', f'Generada (Noche) - Época {epoch}']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(titulos[i])
        plt.imshow(imgs[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'resultados/epoca_{epoch:03d}.png')
    plt.close(fig)

# --- 5. BUCLE DE ENTRENAMIENTO ---
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    gen_g_grads = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    gen_f_grads = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    disc_x_grads = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    disc_y_grads = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    gen_g_opt.apply_gradients(zip(gen_g_grads, generator_g.trainable_variables))
    gen_f_opt.apply_gradients(zip(gen_f_grads, generator_f.trainable_variables))
    disc_x_opt.apply_gradients(zip(disc_x_grads, discriminator_x.trainable_variables))
    disc_y_opt.apply_gradients(zip(disc_y_grads, discriminator_y.trainable_variables))

    return total_gen_g_loss, disc_y_loss

if __name__ == '__main__':
    total_pasos = min(n_train_a, n_train_b)
    muestra_test = next(iter(test_dia))

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        if epoch > 50:
            nuevo_lr = 2e-4 * (1.0 - (epoch - 50) / 30.0)
            gen_g_opt.learning_rate.assign(max(nuevo_lr, 1e-5))
            gen_f_opt.learning_rate.assign(max(nuevo_lr, 1e-5))

        ds_zip = tf.data.Dataset.zip((train_dia, train_noche))
        pbar = tqdm(ds_zip, total=total_pasos, desc=f"Época {epoch}/{EPOCHS}")
        for img_x, img_y in pbar:
            g_loss, d_loss = train_step(img_x, img_y)
            pbar.set_postfix(G=f"{g_loss:.2f}", D=f"{d_loss:.2f}")

        guardar_muestra(generator_g, muestra_test, epoch)

        if epoch % 10 == 0:
            ckpt_manager.save()
            print(f"[Época {epoch}] Checkpoint guardado ({time.time() - start:.1f}s)")

    generator_g.save('generador_dia_noche_final.h5')
    print("Entrenamiento finalizado. Modelo guardado como 'generador_dia_noche_final.h5'")
