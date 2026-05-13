import os
import numpy as np
import torch
import random


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size, device=None):
    """Standard normal noise on the requested device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.normal(mean=0.0, std=1.0, size=size, device=device)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """Embed diffusion steps; keeps tensors on the same device as diffusion_steps."""
    device = diffusion_steps.device
    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim, device=device) * (-_embed))
    _embed = diffusion_steps.float() * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed), torch.cos(_embed)), 2)
    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T, sigma_scorebased=3):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams

def sampling(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=0):
    """
    Perform the complete sampling step according to p(x_0|x_T) = prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                x = x * (1 - mask).float() + cond * mask.float()     # 0's to be imputed, and 1's to preserved
            diffusion_steps = (t * torch.ones((size[0], 1)))  # use the corresponding reverse step
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams):
    """Loss for diffusion model training."""
    B, C, L = X.shape
    device = X.device
    T = diffusion_hyperparams["T"]
    Alpha = diffusion_hyperparams["Alpha"]
    Sigma = diffusion_hyperparams["Sigma"]

    diffusion_steps = torch.randint(T, size=(B, 1, 1), device=device)
    z = std_normal(X.shape, device=device)
    transformed_X = torch.sqrt(Alpha[diffusion_steps]) * X + Sigma[diffusion_steps] * z
    predicted = net(transformed_X, diffusion_steps.view(B, 1), X, None)
    return loss_fn(predicted, z)


def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))  # lenght of series indexes
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        idx = perm[0:k]
        mask[:, channel][idx] = 0

    return mask


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask






###################################
# SCORE BASED
###################################

def compute_true_score(x_t, x_0, alpha_bar_t):
    """
    Compute the true score for Gaussian noise.

    Parameters:
    x_t (torch.Tensor): Noisy data at time t
    x_0 (torch.Tensor): Original data
    alpha_bar_t (torch.Tensor): Cumulative product of alpha up to time t

    Returns:
    torch.Tensor: True score (gradient of log-probability)
    """
    return -(x_t - torch.sqrt(alpha_bar_t) * x_0) / torch.sqrt(1 - alpha_bar_t)

def training_loss_scorebased(net, X, diffusion_hyperparams, loss_fn, only_generate_missing=1):
    """
    Compute the training loss for score-based methods.

    Parameters:
    net (torch.nn.Module): Neural network predicting the score
    X (tuple): Training data and metadata (x_0, cond, mask)
    diffusion_hyperparams (dict): Dictionary of diffusion hyperparameters
    loss_fn (torch.nn.Module): Loss function (e.g., nn.MSELoss)
    only_generate_missing (int): 0 for entire signal, 1 for only missing parts

    Returns:
    torch.Tensor: Loss value
    """
    x_0 = X[0]  # Original data
    cond = X[1]  # Conditional data (if used)
    mask = X[2]  # Mask indicating missing regions

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    B, C, L = x_0.shape

    # Randomly sample diffusion steps
    diffusion_steps = torch.randint(T, size=(B, 1, 1))

    # Generate noisy data x_t
    z = torch.randn_like(x_0)
    x_t = torch.sqrt(Alpha_bar[diffusion_steps]) * x_0 + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z

    # Compute true score
    true_score = compute_true_score(x_t, x_0, Alpha_bar[diffusion_steps])

    # Predict the score using the network
    predicted_score = net((x_t, cond, mask, diffusion_steps.view(B, 1)))

    # Compute loss based on `only_generate_missing`
    if only_generate_missing == 1:
        # Loss for missing parts only
        return loss_fn(predicted_score[mask.bool()], true_score[mask.bool()])
    else:
        # Loss for the entire signal
        return loss_fn(predicted_score, true_score)

def marginal_prob_std(t, sigma, device=None):
    """Marginal std for the SDE (device-safe)."""
    if device is None:
        device = t.device if torch.is_tensor(t) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    return torch.sqrt((sigma**(2 * t) - 1.) / (2. * np.log(sigma)))


def diffusion_coeff(t, sigma, device=None):
    """Diffusion coefficient for the SDE (device-safe)."""
    if device is None:
        device = t.device if torch.is_tensor(t) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    return sigma**t


def training_loss_scorebased_2(model, X, marginal_prob_std_fn, diffusion_hyperparams, only_generate_missing=1, eps=1e-5):

  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  x = X[0] # X = batch, batch, mask, loss mask, 'x' has to be just the batch
  cond = X[1]  # Conditional data (if used)
  mask = X[2]  # Mask indicating missing regions
  #diffusion_steps = torch.randint(T, size=(B, 1, 1))  # randomly sample diffusion steps from 1~T

  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  # this is the noise steps
  z = torch.randn_like(x)
  std = marginal_prob_std_fn(random_t)
  #perturbed_x = x * mask.float() + (z * std[:, None, None])*(1-mask).float()
  perturbed_x = x + z * std[:, None, None] # perturbed_x = x + z * std[:, None, None, None] originally one None more
  random_t = random_t.view(-1,1)

  #net(
  #      (transformed_X, cond, mask, diffusion_steps.view(B, 1),))
  
  score = model((perturbed_x, cond, mask,random_t)) # the mask will be applied to cond in the net forward function
  loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=(1,2)))
  return loss


def Euler_Maruyama_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn,
                        size, mask, cond, eps=1e-3, init_x=None, num_steps=500):
  """Sampler (device-safe). If init_x is provided it will be used."""
  device = cond.device
  batch_size = cond.shape[0]
  if init_x is None:
      x = torch.randn(size, device=device)
  else:
      x = init_x.to(device)

  # Ensure mask/cond shapes broadcast
  x = x * (1 - mask) + cond * mask

  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  for time_step in time_steps:
      batch_time_step = torch.ones(batch_size, 1, device=device) * time_step
      g = diffusion_coeff_fn(batch_time_step)
      x_mean = x + (g**2)[:, None, None] * score_model(x, batch_time_step, cond, mask) * step_size
      x = x_mean + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)
      # keep observed values fixed
      x = x * (1 - mask) + cond * mask
  return x_mean


def sampling_scorebased(net, x_init, diffusion_hyperparams, num_steps,cond, mask):
    """
    Perform sampling using the reverse-time SDE for score-based methods.
    Parameters:
    net (torch.nn.Module): Neural network predicting the score
    x_init (torch.Tensor): Initial noisy sample
    diffusion_hyperparams (dict): Dictionary of diffusion hyperparameters
    num_steps (int): Number of reverse diffusion steps

    Returns:
    torch.Tensor: Reconstructed data
    """
    x = x_init
    T = diffusion_hyperparams["T"]
    Alpha_bar = diffusion_hyperparams["Alpha_bar"]

    with torch.no_grad():
        for t in reversed(range(num_steps)):
            t_tensor = torch.tensor([t])
            alpha_t = Alpha_bar[t_tensor]
            g_t = torch.sqrt(1 - alpha_t)
            # Predict score
            score = net((x, cond, mask, t_tensor)) 

            # Reverse-time drift and diffusion terms
            drift = -(1 - alpha_t) * score
            diffusion = g_t * torch.randn_like(x)

            # Update x
            x = x + drift + diffusion
    return x

