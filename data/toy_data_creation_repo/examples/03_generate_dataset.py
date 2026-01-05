"""
Dataset generation example: Generate tire dataset from configuration.

This example demonstrates:
- Loading configuration from YAML
- Generating dataset with Latin Hypercube Sampling
- Saving images and metadata
"""

from tire_bench.dataset.generator import DatasetGenerator
from pathlib import Path


def main():
    print("=" * 60)
    print("DATASET GENERATION EXAMPLE")
    print("=" * 60)

    # Path to configuration file
    config_path = Path("configs/examples/dataset_generation.yaml")

    if not config_path.exists():
        print(f"\n❌ Config file not found: {config_path}")
        print("   Please run this script from the repository root.")
        return

    # Create generator from config
    print(f"\n1. Loading configuration from {config_path}...")
    generator = DatasetGenerator.from_yaml(str(config_path))

    print(f"   ✓ Configuration loaded")
    print(f"   - Samples to generate: {generator.num_samples}")
    print(f"   - Output directory: {generator.output_dir}")
    print(f"   - Metrics to compute: {[m.name for m in generator.metrics]}")

    # Generate dataset
    print("\n2. Generating dataset...")
    print("   (This may take a while depending on num_samples)")

    dataset = generator.generate()

    # Show summary
    print("\n3. Dataset summary:")
    print(f"   - Total samples: {len(dataset)}")
    print(f"   - Columns: {list(dataset.columns)}")

    # Show statistics
    if "K_vert" in dataset.columns:
        print(f"\n   K_vert statistics:")
        print(f"   - Mean: {dataset['K_vert'].mean():.2f}")
        print(f"   - Std:  {dataset['K_vert'].std():.2f}")
        print(f"   - Min:  {dataset['K_vert'].min():.2f}")
        print(f"   - Max:  {dataset['K_vert'].max():.2f}")

    if "performance_index" in dataset.columns:
        print(f"\n   Performance index statistics:")
        print(f"   - Mean: {dataset['performance_index'].mean():.5f}")
        print(f"   - Std:  {dataset['performance_index'].std():.5f}")
        print(f"   - Min:  {dataset['performance_index'].min():.5f}")
        print(f"   - Max:  {dataset['performance_index'].max():.5f}")

    # Show sample
    print("\n4. First sample:")
    print(dataset.head(1).T)

    print("\n" + "=" * 60)
    print("✓ Dataset generation completed successfully!")
    print(f"\n   Dataset saved to: {generator.output_dir}")
    print(f"   - Metadata: metadata.csv")
    print(f"   - Images: sample_XXXXX/ directories")
    print("=" * 60)


if __name__ == "__main__":
    main()
