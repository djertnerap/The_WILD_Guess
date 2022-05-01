#
# An extension of DORO to groupDORO
#
import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from wilds.common.utils import get_counts

import math
import scipy.optimize as sopt
import torch.nn.functional as F

class GroupDORO(SingleModelAlgorithm):
    """
    groupDORO algorithm
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, is_group_in_train):
        # check config
        assert config.uniform_over_groups
        # initialize model
        model = initialize_model(config, d_out)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # additional logging
        self.logged_fields.append('group_weight')
        # step size
        self.group_weights_step_size = config.group_doro_step_size
        # initialize adversarial weights
        self.group_weights = torch.zeros(grouper.n_groups)
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
        self.group_weights = self.group_weights.to(self.device)
        
        # initialize doro parameters
        self.alpha = config.alpha
        self.eps = config.eps
        self.alg = config.doro_alg

    def process_batch(self, batch, unlabeled_batch=None):
        results = super().process_batch(batch)
        results['group_weight'] = self.group_weights
        return results

    def cvar_doro(self, t):
        batch_size = len(t)
        gamma = self.eps + self.alpha * (1 - self.eps)
        n1 = int(gamma * batch_size)
        n2 = int(self.eps * batch_size)
        rk = torch.argsort(t, descending=True)
        #print(f"Before CVAR Filter: {batch_size}, After: {t[rk[n2:n1]].size()}")
        loss = t[rk[n2:n1]].sum() / self.alpha / (batch_size - n2)
        return loss

    def objective(self, results):
        """
        Takes an output of SingleModelAlgorithm.process_batch() and computes the
        optimized objective. For group DRO, the objective is the weighted average
        of losses, where groups have weights groupDRO.group_weights.
        Args:
            - results (dictionary): output of SingleModelAlgorithm.process_batch()
        Output:
            - objective (Tensor): optimized objective; size (1,).
        """
        loss = self.loss.compute_element_wise(
            results['y_pred'],
            results['y_true'],
            return_dict=False
        )

        # Group losses by region and run cvar_doro
        group_counts = get_counts(results['g'], self.grouper.n_groups)
        group_metrics = []
        for group_idx in range(self.grouper.n_groups):
            if group_counts[group_idx]==0:
                group_metrics.append(torch.tensor(0., device=results['g'].device))
            else:
                group_metrics.append(
                    self.cvar_doro(
                        loss[results['g'] == group_idx]
                    )
                )
        group_losses = torch.stack(group_metrics)

        return group_losses @ self.group_weights

    def _update(self, results, should_step=True):
        """
        Process the batch, update the log, and update the model, group weights, and scheduler.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
                - objective (float)
        """
        # compute group losses
        loss = self.loss.compute_element_wise(
            results['y_pred'],
            results['y_true'],
            return_dict=False
        )

        # Group losses by region and run cvar_doro
        group_counts = get_counts(results['g'], self.grouper.n_groups)
        group_metrics = []
        for group_idx in range(self.grouper.n_groups):
            if group_counts[group_idx]==0:
                group_metrics.append(torch.tensor(0., device=results['g'].device))
            else:
                group_metrics.append(
                    self.cvar_doro(
                        loss[results['g'] == group_idx]
                    )
                )
        group_losses = torch.stack(group_metrics)

        # update group weights
        self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size*group_losses.data)
        self.group_weights = (self.group_weights/(self.group_weights.sum()))
        # save updated group weights
        results['group_weight'] = self.group_weights
        # update model
        super()._update(results, should_step=should_step)