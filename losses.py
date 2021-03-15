import torch
import numpy as np
import math


def neg_multi_log_likelihood(gt, pred, confidences, avails, reduce_mean=True):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    if not torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))):
        print(confidences)
    assert torch.allclose(torch.sum(confidences, dim=1),
                          confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    # error (batch_size, num_modes)
    error = confidences + 1e-16 - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=1, keepdim=True)

    # print("error", error)
    if reduce_mean:
        return torch.mean(error)
    else:
        return error


def neg_multi_log_likelihood_single(gt, pred, avails):
    """
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return neg_multi_log_likelihood(gt, pred.unsqueeze(1), confidences, avails)


def custom_angle_loss(gt, pred, avails, device, penalize=1.25):
    """
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: array of shape (bs)x(time), custom parameter to penalize more on corner cases.
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)

    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))

    y_tensor = torch.tensor((), dtype=torch.float64)
    yaws = y_tensor.new_zeros((batch_size,), requires_grad=False)
    for i in range(batch_size):
        # compare first
        sing_avail = avails[i]
        sing_targets = gt[i]
        ones = (sing_avail == 1).nonzero()
        first, last = ones[0], ones[-1]
        # print(first, last)
        dx = sing_targets[last][0][0].item() - sing_targets[first][0][0].item()
        dy = sing_targets[last][0][1].item() - sing_targets[first][0][1].item()

        yaw_agent = math.acos(abs(dx) / (math.sqrt(dx ** 2 + dy ** 2))) * penalize
        yaw_agent = 0.5 if yaw_agent < 0.5 else yaw_agent
        yaws[i] = yaw_agent

    yaws.to(device).detach()

    return (neg_multi_log_likelihood(gt, pred.unsqueeze(1), confidences, avails) * yaws).mean()