
import builtins
from typing import Union, Callable

from torch import Tensor, einsum

from .utils import is_scalar, is_tensor, get_coef_subscripts


Number = Union[builtins.int, builtins.float]
CoefLike = Union[Number, Tensor, Callable[..., Tensor]]


def linear_integral(input: Tensor, weights: Tensor, measure: Tensor,
                    coef: Union[Number, Tensor, None]=None,
                    batched: bool=False) -> Tensor:
    r"""Numerical integration.

    Args:
        input (Tensor[Q, C, I]): The values on the quadrature points to be integrated.
        weights (Tensor[Q,]): The weights of the quadrature points.
        measure (Tensor[C,]): The measure of the quadrature points.
        coef (Number, Tensor, optional): The coefficient of the integration. Defaults to None.
        Must be int, float, Tensor, or callable returning Tensor with shape (Q,), (C,) or (Q, C).
        If `batched == True`, the shape of the coef should be (Q, C, B) or (C, B).
        batched (bool, optional): Whether the coef are batched. Defaults to False.

    Returns:
        Tensor[C, I]: The result of the integration.
        If `batched == True`, the shape of the result is (C, I, B).
    """
    if coef is None:
        return einsum('q, c, qci -> ci', weights, measure, input)

    NQ = weights.shape[0]
    NC = measure.shape[0]

    if is_scalar(coef):
        return einsum('q, c, qci -> ci', weights, measure, input) * coef
    elif is_tensor(coef):
        out_subs = 'cib' if batched else 'ci'
        subs = get_coef_subscripts(coef.shape, NQ, NC, batched)
        return einsum(f'q, c, qci, {subs} -> {out_subs}', weights, measure, input, coef)
    else:
        raise TypeError(f"coef should be int, float, Tensor or callable, but got {type(coef)}.")


def bilinear_integral(input1: Tensor, input2: Tensor, weights: Tensor, measure: Tensor,
                      coef: Union[Number, Tensor, None]=None,
                      batched: bool=False) -> Tensor:
    r"""Numerical integration.

    Args:
        input1 (Tensor[Q, C, I, ...]): The values on the quadrature points to be integrated.
        input2 (Tensor[Q, C, J, ...]): The values on the quadrature points to be integrated.
        weights (Tensor[Q,]): The weights of the quadrature points.
        measure (Tensor[C,]): The measure of the quadrature points.
        coef (Number, Tensor, optional): The coefficient of the integration. Defaults to None.
        Must be int, float, Tensor, or callable returning Tensor with shape (Q,), (C,) or (Q, C).
        If `batched == True`, the shape of the coef should be (Q, C, B) or (C, B).
        batched (bool, optional): Whether the coef are batched. Defaults to False.

    Returns:
        Tensor[C, I, J]: The result of the integration.
        If `batched == True`, the shape of the output is (C, I, J, B).
    """
    if coef is None:
        return einsum('q, c, qci..., qcj... -> cij', weights, measure, input1, input2)

    NQ = weights.shape[0]
    NC = measure.shape[0]

    if is_scalar(coef):
        return einsum('q, c, qci..., qcj... -> cij', weights, measure, input1, input2) * coef
    elif is_tensor(coef):
        out_subs = 'cijb' if batched else 'cij'
        subs = get_coef_subscripts(coef.shape, NQ, NC, batched)
        return einsum(f'q, c, qci..., qcj..., {subs} -> {out_subs}', weights, measure, input1, input2, coef)
    else:
        raise TypeError(f"coef should be int, float, Tensor or callable, but got {type(coef)}.")