"""Loss functions."""

# FIXME

import functools
import inspect
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_LOSS_FUNCTIONS = {}

Array = torch.Tensor
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]
ArrayDict = Dict[str, Array]
DictTree = Dict[str, Union[Array, "DictTree"]]
LossFn = Callable[[Dict[str, ArrayTree], Dict[str, ArrayTree]],
				  Tuple[Array, ArrayTree]]
ConfigAttr = Any
MetricSpec = Dict[str, str]


def standardize_loss_config(
	loss_config: Union[Sequence[str], Dict]
) -> Dict:
	"""Standardize loss configs into a common Dict format.

	Args:
		loss_config: List of strings or Dict specifying loss configuration.
			Valid input formats are:
				Option 1 (list of strings):
					ex) `["box", "presence"]`
				Option 2 (losses with weights only):
					ex) `{"box": 5, "presence": 2}`
				Option 3 (losses with weights and other parameters):
					ex) `{"box": {"weight" 5, "metric": "l1"}, "presence": {"weight": 2}}

	Returns:
		Standardized Dict containing the loss configuration

	Raises:
		ValueError: If loss_config is a list that contains non-string entries.
	"""
	
	if isinstance(loss_config, Sequence): # Option 1
		if not all(isinstance(loss_type, str) for loss_type in loss_config):
			raise ValueError(f"Loss types all need to be str but got {loss_config}")
		return {k: {} for k in loss_config}

	# Convert all option-2-style weights to option-3-style dictionaries.
	if not isinstance(loss_config, Dict):
		raise ValueError(f"Loss config type not Sequence or Dict; got {loss_config}")
	else:
		loss_config = {
			k: {
				"weight": v
			} if isinstance(v, (float, int)) else v for k, v in loss_config.items()
		}
	return loss_config


def update_loss_aux(loss_aux: Dict[str, Array], update: Dict[str, Array]):
	existing_keys = set(update.keys()).intersection(loss_aux.keys())
	if existing_keys:
		raise KeyError(
			f"Can't overwrite existing keys in loss_aux: {existing_keys}")
	loss_aux.update(update)


def compute_full_loss(
	preds: Dict[str, ArrayTree], targets: Dict[str, ArrayTree],
	loss_config: Union[Sequence[str], Dict]
) -> Tuple[Array, ArrayTree]:
	"""Loss function that parses and combines weighted loss terms.

	Args:
		preds: Dictionary of tensors containing model predictions.
		targets: Dictionary of tensors containing prediction targets.
		loss_config: List of strings or Dict specifying loss configuration.
			See @register_loss decorated functions below for valid loss names.
			Valid losses formats are:
				- Option 1 (list of strings):
					ex) `["box", "presence"]`
				- Option 2 (losses with weights only):
					ex) `{"box": 5, "presence": 2}`
				- Option 3 (losses with weights and other parameters)
					ex) `{"box": {"weight": 5, "metric": "l1}, "presence": {"weight": 2}}`
				- Option 4 (like 3 but decoupling name and loss_type)
					ex) `{"recon_flow": {"loss_type": "recon", "key": "flow"},
						  "recon_video": {"loss_type": "recon", "key": "video"}}`   

	Returns:
		A 2-tuple of the sum of all individual loss terms and a dictionary of
			auxiliary losses and metrics.
	"""

	loss = torch.zeros_like(torch.Tensor(), dtype=torch.float32)
	loss_aux = {}
	loss_config = standardize_loss_config(loss_config)
	for loss_name, cfg in loss_config.items():
		context_kwargs = {"preds": preds, "targets": targets}
		weight, loss_term, loss_aux_update = comput_loss_term(
			loss_name=loss_name, context_kwargs=context_kwargs, config_kwargs=cfg)

		unweighted_loss = torch.mean(loss_term)
		loss += weight * unweighted_loss
		loss_aux_update[loss_name + "_value"] = unweighted_loss
		loss_aux_update[loss_name + "_weight"] = torch.ones_like(unweighted_loss)
		update_loss_aux(loss_aux, loss_aux_update)
	return loss, loss_aux


def register_loss(func=None,
				  *,
				  name: Optional[str] = None,
				  check_unused_kwargs: bool = True):
		"""Decorator for registering a loss function.

		Can be used without arguments:
		```
		@register_loss
		def my_loss(**_):
			return 0
		```
		or with keyword arguments:
		```
		@register_loss(name="my_renamed_loss")
		def my_loss(**_):
			return 0
		```

		Loss functions may accept
			- context kwargs: `preds` and `targets`
			- config kwargs: any argument specified in the config
			- the special `config_kwargs` parameter that contains the entire loss config.
		Loss functions also _need_ to accept a **kwarg argument to support extending
		the interface.
		They should return either:
			- just the computed loss (pre-reduction)
			- or a tuple of the computed loss and a loss_aux_updates dict
		
		Args:
			func: the decorated function
			name (str): optional name to be used for this loss in the config.
				Defaults to the name of the function.
			check_unused_kwargs (bool): By default compute_loss_term raises an error if
				there are any usused config kwargs. If this flag is set to False that step
				is skipped. This is useful if the config_kwargs should be passed onward to
				another function.

		Returns:
			The decorated function (or a partial of the decorator)
		"""
		# If this decorator has been called with parameters but no function, then we
		# return the decorator again (but with partially filled parameters).
		# This allows using both @register_loss and @register_loss(name="foo")
		if func is None:
			return functools.partial(
				register_loss, name=name, check_unused_kwargs=check_unused_kwargs)

		# No (further) arguments: this is the actual decorator
		# ensure that the loss function includes a **kwargs argument
		loss_name = name if name is not None else func.__name__
		if not any(v.kind == inspect.Parameter.VAR_KEYWORD
				   for k, v in inspect.signature(func).parameters.items()):
			raise TypeError(
				f"Loss function '{loss_name}' needs to include a **kwargs argument")
		func.name = loss_name
		func.check_unused_kwargs = check_unused_kwargs
		_LOSS_FUNCTIONS[loss_name] = func
		return func


def compute_loss_term(
	loss_name: str, context_kwargs: Dict[str, Any],
	config_kwargs: Dict[str, Any]) -> Tuple[float, Array, Dict[str, Array]]:
	"""Compute a loss function given its config and context parameters.

	Takes care of:
		- finding the correct loss function based on "loss_type" or name
		- the optional "weight" parameter
		- checking for typos and collisions in config parameters
		- adding the optional loss_aux_updates if omitted by the loss_fn

	Args:
		loss_name: Name of the loss, i.e. its key in the config.losses dict.
		context_kwargs: Dictionary of context variables (`preds` and `targets`)
		config_kwargs: The config dict for this loss

	Returns:
		1. the loss weight (float)
		2. loss term (Array)
		3. loss aux updates (Dict[str, Array])

	Raises:
		KeyError:
			Unknown loss_type
		KeyError:
			Unused config entries, i.e. not used by the loss function.
			Not raised if using @reegister_loss(check_unused_kwargs=False)
		KeyError: Config entry with a name that conflicts with a context_kwarg
		ValueError: Non-numerical weight in config_kwargs
	"""

	# Make a dict copy of config_kwargs
	kwargs = {k: v for k, v in config_kwargs.items()}

	# Get the loss function
	loss_type = kwargs.pop("loss_type", loss_name)
	if loss_type not in _LOSS_FUNCTIONS:
		raise KeyError(f"Unknown loss_type '{loss_type}'.")
	loss_fn = _LOSS_FUNCTIONS[loss_type]

	# Take care of "weight" term
	weight = kwargs.pop("weight", 1.0)
	if not isinstance(weight, (int, float)):
		raise ValueError(f"Weight for loss {loss_name} should be a number, "
						 f"but was {weight}.")

	# Check for unused config entries (to prevent typos etc.)
	config_keys = set(kwargs)
	if loss_fn.check_unused_kwargs:
		param_names = set(inspect.signature(loss_fn).parameters)
		unused_config_keys = config_keys - param_names
		if unused_config_keys:
			raise KeyError(f"Unrecognized config entries {unused_config_keys} "
						   f"for loss {loss_name}.")

	# Check for key collisions between context and config
	conflicting_config_keys = config_keys.intersection(context_kwargs)
	if conflicting_config_keys:
		raise KeyError(f"The config keys {conflicting_config_keys} conflict "
					f"with the context parameters ({context_kwargs.keys()}) "
					f"for loss {loss_name}.")

	# Construct the arguments for the loss function
	kwargs.update(context_kwargs)
	kwargs["config_kwargs"] = config_kwargs

	# Call loss
	results = loss_fn(**kwargs)

	# Add empty loss_aux_updates if neceessary
	if isinstance(results, Tuple):
		loss, loss_aux_update = results
	else:
		loss, loss_aux_update = results, {}

	return weight, loss, loss_aux_update


# -------- Loss functions --------
@register_loss
def recon(preds: ArrayTree,
		  targets: ArrayTree,
		  key: str = "video",
		  reduction_type: str = "sum",
		  **_) -> float:
	"""Reconstruction loss (MSE)."""
	inputs = preds["outputs"][key]
	targets = targets[key]
	loss = F.mse_loss(inputs, targets, reduction=reduction_type)
	if reduction_type == "mean":
		# This rescaling reflects taking the sum over feature axis &
		# mean over space/time axis
		loss *= targets.shape[-1]
	return torch.mean(loss)

def recon_loss(preds: ArrayTree,
			   targets: ArrayTree,
			   reduction_type: str = "sum") -> float:
	"""Reconstruction loss (MSE)."""
	inputs = preds
	targets = targets
	loss = F.mse_loss(inputs, targets, reduction=reduction_type)
	if reduction_type == "mean":
		# This rescaling reflects taking the sum over feature axis &
		# mean over space/time axis
		loss *= targets.shape[-1]
	return loss.mean()

class Recon_Loss(nn.Module):

	def __init__(self):
		super().__init__()
		self.l2 = nn.MSELoss(reduction="sum")
	
	def forward(self, inputs, targets):
		# print('in, tar', inputs.shape, targets.shape)

		loss = self.l2(inputs, targets)
		return torch.mean(loss)

# def squared_l2_norm(preds: Array, targets: Array,
# 					reduction_type: str = "sum") -> Array:
# 	"""Squared L2 norm.
# 	reduction: in `["sum", "mean"]`
# 	"""
# 	if reduction_type =="sum"