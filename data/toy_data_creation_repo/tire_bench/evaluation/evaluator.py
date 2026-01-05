"""Deformation evaluation for generative models."""

from typing import Dict
import numpy as np

from tire_bench.core.tire import Tire
from tire_bench.mechanics.rigid_ring import RigidRingModel


class DeformationEvaluator:
    """
    Evaluate predicted deformations against theoretical models.

    For generative models that predict deformed tire from original tire.

    Evaluation metrics:
    - Volume conservation (tire volume should be preserved)
    - Shape similarity (IoU with theoretical deformation)
    - Component ratio preservation (relative sizes of components)
    - Height change accuracy (deflection δ)
    """

    def __init__(self, mechanics_model: str = "rigid_ring"):
        """
        Initialize evaluator.

        Args:
            mechanics_model: Deformation model to use ('rigid_ring')
        """
        if mechanics_model == "rigid_ring":
            self.model = RigidRingModel()
        else:
            raise ValueError(f"Unknown mechanics model: {mechanics_model}")

    def evaluate(
        self, original_tire: Tire, predicted_tire: Tire, force: float
    ) -> Dict[str, float]:
        """
        Evaluate prediction quality.

        Args:
            original_tire: Original tire
            predicted_tire: Predicted deformed tire
            force: Applied force

        Returns:
            Dict of evaluation scores (0-1, higher is better)
        """
        # Compute theoretical deformation
        result = self.model.apply_load(original_tire, force)
        theoretical_tire = result["deformed_tire"]

        scores = {}

        # 1. Volume conservation
        scores["volume_conservation"] = self._compute_volume_conservation(
            original_tire, predicted_tire
        )

        # 2. Shape similarity (IoU with theoretical)
        scores["shape_similarity"] = self._compute_shape_similarity(
            predicted_tire, theoretical_tire
        )

        # 3. Component ratio preservation
        scores["component_ratio"] = self._compute_ratio_preservation(
            original_tire, predicted_tire
        )

        # 4. Height change accuracy
        scores["height_accuracy"] = self._compute_height_accuracy(
            original_tire, predicted_tire, theoretical_tire
        )

        # Global score (weighted average)
        weights = {
            "volume_conservation": 0.25,
            "shape_similarity": 0.35,
            "component_ratio": 0.20,
            "height_accuracy": 0.20,
        }
        scores["global_score"] = sum(weights[k] * scores[k] for k in weights)

        return scores

    def _compute_volume_conservation(
        self, original_tire: Tire, predicted_tire: Tire
    ) -> float:
        """
        Check volume conservation.

        Rubber is nearly incompressible, so total volume should be preserved.

        Returns:
            Score in [0, 1], 1 = perfect conservation
        """
        orig_vol = sum(c.image.sum() for c in original_tire.components.values())
        pred_vol = sum(c.image.sum() for c in predicted_tire.components.values())

        if orig_vol == 0:
            return 0.0

        ratio = pred_vol / orig_vol
        # Penalize deviation from 1.0
        score = np.exp(-5 * abs(ratio - 1.0))

        return float(score)

    def _compute_shape_similarity(
        self, predicted_tire: Tire, theoretical_tire: Tire
    ) -> float:
        """
        Compute shape similarity using IoU (Intersection over Union).

        Returns:
            Average IoU across all components
        """
        ious = []

        for name in predicted_tire.component_names:
            if name not in theoretical_tire.component_names:
                continue

            pred_img = predicted_tire.get_component(name).image
            theo_img = theoretical_tire.get_component(name).image

            iou = self._compute_iou(pred_img, theo_img)
            ious.append(iou)

        return float(np.mean(ious)) if ious else 0.0

    def _compute_iou(self, pred: np.ndarray, theo: np.ndarray) -> float:
        """Compute Intersection over Union."""
        pred_bin = pred > 0.5
        theo_bin = theo > 0.5

        intersection = (pred_bin & theo_bin).sum()
        union = (pred_bin | theo_bin).sum()

        return intersection / union if union > 0 else 0.0

    def _compute_ratio_preservation(
        self, original_tire: Tire, predicted_tire: Tire
    ) -> float:
        """
        Check if relative component sizes are preserved.

        Returns:
            Score in [0, 1], 1 = perfect preservation
        """
        orig_vols = {
            name: comp.image.sum() for name, comp in original_tire.components.items()
        }
        pred_vols = {
            name: comp.image.sum() for name, comp in predicted_tire.components.items()
        }

        total_orig = sum(orig_vols.values())
        total_pred = sum(pred_vols.values())

        if total_orig == 0 or total_pred == 0:
            return 0.0

        # Compute ratio differences
        diffs = []
        for name in orig_vols:
            if name not in pred_vols:
                continue

            orig_ratio = orig_vols[name] / total_orig
            pred_ratio = pred_vols[name] / total_pred

            diff = abs(orig_ratio - pred_ratio)
            diffs.append(diff)

        if not diffs:
            return 0.0

        # Average difference, convert to score
        avg_diff = np.mean(diffs)
        score = np.exp(-10 * avg_diff)

        return float(score)

    def _compute_height_accuracy(
        self, original_tire: Tire, predicted_tire: Tire, theoretical_tire: Tire
    ) -> float:
        """
        Check if height change matches theoretical prediction.

        Returns:
            Score in [0, 1], 1 = perfect match
        """
        # Compute heights
        orig_height = self._get_tire_height(original_tire)
        pred_height = self._get_tire_height(predicted_tire)
        theo_height = self._get_tire_height(theoretical_tire)

        if orig_height == 0:
            return 0.0

        # Compute deflections
        pred_delta = orig_height - pred_height
        theo_delta = orig_height - theo_height

        if theo_delta == 0:
            return 1.0 if pred_delta == 0 else 0.0

        # Relative error
        rel_error = abs(pred_delta - theo_delta) / theo_delta
        score = np.exp(-2 * rel_error)

        return float(score)

    def _get_tire_height(self, tire: Tire) -> float:
        """Get tire height in pixels."""
        full_mask = tire.get_full_mask()

        if full_mask.sum() == 0:
            return 0.0

        rows = np.where(full_mask.any(axis=1))[0]
        height = rows.max() - rows.min() + 1

        return float(height)
