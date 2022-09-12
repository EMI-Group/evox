from flax import linen as nn

class LetNet(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=6, kernel_size=(5, 5), padding='SAME')(x)
    x = nn.sigmoid(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=16, kernel_size=(5, 5), padding='SAME')(x)
    x = nn.sigmoid(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape(x.shape[0], -1)
    x = nn.Dense(120)(x)
    x = nn.sigmoid(x)
    x = nn.Dense(84)(x)
    x = nn.sigmoid(x)
    x = nn.Dense(10)(x)
    return x


class SimpleCNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=6, kernel_size=(5, 5), padding='SAME')(x)
    x = nn.sigmoid(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=16, kernel_size=(5, 5), padding='SAME')(x)
    x = nn.sigmoid(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape(x.shape[0], -1)
    x = nn.Dense(120)(x)
    x = nn.sigmoid(x)
    x = nn.Dense(10)(x)
    return x