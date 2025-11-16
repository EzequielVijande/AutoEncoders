# factory.py

from enum import Enum, auto
from typing import Dict, Type, Union
from .base import ActivationFunction
from .step import StepActivation
from .step_bipolar import StepBipolarActivation
from .sigmoid import SigmoidActivation
from .relu import ReLUActivation
from .linear import LinearActivation
from .tanh import TanhActivation


class ActivationFunctionType(Enum):
    """Enum para tipos de funciones de activación"""
    STEP = auto()
    STEP_BIPOLAR = auto()
    SIGMOID = auto()
    RELU = auto()
    TANH = auto()
    LINEAR = auto()


class ActivationFunctionFactory:
    # Mapeo de enum a clases
    _ENUM_TO_CLASS: Dict[ActivationFunctionType, Type[ActivationFunction]] = {
        ActivationFunctionType.STEP: StepActivation,
        ActivationFunctionType.STEP_BIPOLAR: StepBipolarActivation,
        ActivationFunctionType.SIGMOID: SigmoidActivation,
        ActivationFunctionType.RELU: ReLUActivation,
        ActivationFunctionType.TANH: TanhActivation,
        ActivationFunctionType.LINEAR: LinearActivation,
    }
    
    # Mapeo de string a enum
    _STRING_TO_ENUM: Dict[str, ActivationFunctionType] = {
        "STEP": ActivationFunctionType.STEP,
        "STEP_BIPOLAR": ActivationFunctionType.STEP_BIPOLAR,
        "SIGMOID": ActivationFunctionType.SIGMOID,
        "RELU": ActivationFunctionType.RELU,
        "TANH": ActivationFunctionType.TANH,
        "LINEAR": ActivationFunctionType.LINEAR,
    }

    @classmethod
    def create(cls, activation_type: str) -> ActivationFunction:
        activation_type = activation_type.upper()
        
        if activation_type not in cls._STRING_TO_ENUM:
            available_types = list(cls._STRING_TO_ENUM.keys())
            raise ValueError(
                f"Tipo de función de activación '{activation_type}' no soportado. "
                f"Tipos disponibles: {available_types}"
            )
        
        enum_type = cls._STRING_TO_ENUM[activation_type]
        function_class = cls._ENUM_TO_CLASS[enum_type]
        return function_class()
    
    @classmethod
    def create_from_enum(cls, activation_type: ActivationFunctionType) -> ActivationFunction:
        if activation_type not in cls._ENUM_TO_CLASS:
            available_types = list(cls._ENUM_TO_CLASS.keys())
            raise ValueError(
                f"Tipo de función de activación '{activation_type}' no soportado. "
                f"Tipos disponibles: {available_types}"
            )
        
        function_class = cls._ENUM_TO_CLASS[activation_type]
        return function_class()
    
    @classmethod
    def get_available_types(cls) -> list:
        """Retorna la lista de tipos de activación disponibles como strings"""
        return list(cls._STRING_TO_ENUM.keys())
    
    @classmethod
    def get_available_enums(cls) -> list:
        """Retorna la lista de tipos de activación disponibles como enums"""
        return list(cls._ENUM_TO_CLASS.keys())


