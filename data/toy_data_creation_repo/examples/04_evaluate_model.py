"""
Model evaluation example: Evaluate predicted tire deformations.

This example demonstrates:
- Using the DeformationEvaluator
- Comparing predicted vs theoretical deformations
- Computing evaluation scores
"""

from tire_bench.core.tire import Tire
from tire_bench.core.materials import MaterialProperties
from tire_bench.geometry.generator import (
    CarcassComponent,
    CrownComponent,
    FlanksComponent,
)
from tire_bench.mechanics.rigid_ring import RigidRingModel
from tire_bench.evaluation.evaluator import DeformationEvaluator


def create_sample_tire():
    """Create a sample tire for testing."""
    tire = Tire(resolution=64)

    mat_carcass = MaterialProperties(E=1.0, rho=1.0, name="carcass")
    mat_crown = MaterialProperties(E=0.8, rho=1.2, name="crown")
    mat_flanks = MaterialProperties(E=0.5, rho=0.8, name="flanks")

    tire.add_component(
        "carcass",
        CarcassComponent(
            name="carcass", material=mat_carcass, resolution=64, thickness=4, w_belly=24
        ),
    )
    tire.add_component(
        "crown",
        CrownComponent(
            name="crown",
            material=mat_crown,
            resolution=64,
            thickness=5,
            thickness_carcass=4,
            w_belly=24,
        ),
    )
    tire.add_component(
        "flanks",
        FlanksComponent(
            name="flanks",
            material=mat_flanks,
            resolution=64,
            thickness_carcass=4,
            thickness_crown=5,
            w_belly=24,
        ),
    )

    return tire


def main():
    print("=" * 60)
    print("MODEL EVALUATION EXAMPLE")
    print("=" * 60)

    # 1. Create original tire
    print("\n1. Creating original tire...")
    original_tire = create_sample_tire()
    print("   ✓ Original tire created")

    # 2. Generate "prediction" (using theoretical model with some noise)
    print("\n2. Generating predicted deformation...")
    print("   (In practice, this would come from your generative model)")

    mechanics = RigidRingModel()
    force = 200

    # Get theoretical deformation
    result = mechanics.apply_load(original_tire, force)
    theoretical_tire = result["deformed_tire"]

    # For this example, we'll use the theoretical as "prediction"
    # In practice, you'd load predictions from your model
    predicted_tire = theoretical_tire

    print(f"   ✓ Predicted deformation at F={force}")

    # 3. Evaluate prediction
    print("\n3. Evaluating prediction quality...")
    evaluator = DeformationEvaluator(mechanics_model="rigid_ring")

    scores = evaluator.evaluate(original_tire, predicted_tire, force)

    print("\n   Evaluation scores (0-1, higher is better):")
    print("   " + "-" * 45)
    print(f"   Volume conservation  : {scores['volume_conservation']:.4f}")
    print(f"   Shape similarity     : {scores['shape_similarity']:.4f}")
    print(f"   Component ratio      : {scores['component_ratio']:.4f}")
    print(f"   Height accuracy      : {scores['height_accuracy']:.4f}")
    print("   " + "-" * 45)
    print(f"   GLOBAL SCORE         : {scores['global_score']:.4f}")
    print("   " + "-" * 45)

    # 4. Interpretation
    print("\n4. Score interpretation:")

    global_score = scores["global_score"]
    if global_score > 0.9:
        quality = "Excellent"
    elif global_score > 0.7:
        quality = "Good"
    elif global_score > 0.5:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"   Prediction quality: {quality}")

    print("\n   Individual score meanings:")
    print("   - Volume conservation: Tire volume preserved (incompressibility)")
    print("   - Shape similarity: IoU with theoretical deformation")
    print("   - Component ratio: Relative component sizes maintained")
    print("   - Height accuracy: Deflection δ matches theoretical")

    print("\n" + "=" * 60)
    print("✓ Evaluation example completed!")
    print("\nUse case:")
    print("  Evaluate generative models that predict tire deformation")
    print("  from input tire geometry and applied force.")
    print("=" * 60)


if __name__ == "__main__":
    main()
