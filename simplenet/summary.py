from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# To read the model parameters

def focal_loss(gamma=2.0, alpha=0.5):
    def loss_fn(y_true, y_pred):
        
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        
        return tf.reduce_mean(weight * cross_entropy)
    
    return loss_fn

model_path = '../model/retina_model.h5'
with custom_object_scope({'focal_loss': focal_loss, 'loss_fn': focal_loss()}):
    model = load_model(model_path)
model.summary()

for i, layer in enumerate(model.layers):
    print(f"\nLayer {i}: {layer.name}")
    print(f"Input shape: {layer.input_shape}")
    print(f"Output shape: {layer.output_shape}")
    print(f"Trainable weights: {len(layer.trainable_weights)}")