"""
Quick validation script to test the implementation.
"""

print("Testing Tire Deformation Benchmark Implementation...")
print("=" * 60)

# Test 1: Imports
print("\n1. Testing imports...")
try:
    from tire_bench import Tire, MaterialProperties, ComponentBase, ComponentRegistry
    from tire_bench.geometry import CarcassComponent, CrownComponent, FlanksComponent
    from tire_bench.mechanics import RigidRingModel
    from tire_bench.metrics import MetricRegistry
    from tire_bench.metrics.stiffness import VerticalStiffnessMetric
    from tire_bench.metrics.mass import MassMetric
    from tire_bench.metrics.performance import PerformanceRatioMetric
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Test 2: Create tire
print("\n2. Creating tire with components...")
try:
    tire = Tire(resolution=64)

    mat_carcass = MaterialProperties(E=1.0, rho=1.0, name="carcass")
    mat_crown = MaterialProperties(E=0.8, rho=1.2, name="crown")
    mat_flanks = MaterialProperties(E=0.5, rho=0.8, name="flanks")

    carcass = CarcassComponent(
        name="carcass",
        material=mat_carcass,
        resolution=64,
        thickness=4,
        w_belly=24
    )

    crown = CrownComponent(
        name="crown",
        material=mat_crown,
        resolution=64,
        thickness=5,
        thickness_carcass=4,
        w_belly=24
    )

    flanks = FlanksComponent(
        name="flanks",
        material=mat_flanks,
        resolution=64,
        thickness_carcass=4,
        thickness_crown=5,
        w_belly=24
    )

    tire.add_component("carcass", carcass)
    tire.add_component("crown", crown)
    tire.add_component("flanks", flanks)

    print(f"   ✓ Tire created with {len(tire.components)} components")
except Exception as e:
    print(f"   ❌ Tire creation error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Compute properties
print("\n3. Computing mechanical properties...")
try:
    mechanics = RigidRingModel()
    props = mechanics.compute_properties(tire)

    print(f"   K_vert: {props['K_vert']:.2f}")
    print(f"   Mass: {props['mass_index']:.0f}")
    print(f"   Performance: {props['performance_index']:.4f}")

    # Validate results are in reasonable range
    assert props['K_vert'] > 0, "K_vert should be positive"
    assert props['mass_index'] > 0, "Mass should be positive"
    assert props['performance_index'] > 0, "Performance should be positive"

    print("   ✓ Properties computed successfully")
except Exception as e:
    print(f"   ❌ Property computation error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Apply deformation
print("\n4. Applying deformation...")
try:
    result = mechanics.apply_load(tire, force=200)

    print(f"   Deflection δ: {result['delta']:.1f} px")
    print(f"   K_vert: {result['K_vert']:.2f}")

    # Validate deformed tire
    deformed_tire = result['deformed_tire']
    assert isinstance(deformed_tire, Tire), "Deformed tire should be Tire instance"
    assert len(deformed_tire.components) == len(tire.components), "Should have same components"

    print("   ✓ Deformation applied successfully")
except Exception as e:
    print(f"   ❌ Deformation error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test metrics
print("\n5. Testing metrics system...")
try:
    # Test metric registry
    available_metrics = MetricRegistry.list_metrics()
    print(f"   Available metrics: {available_metrics}")

    # Create and use metrics
    stiffness = MetricRegistry.create('vertical_stiffness')
    mass = MetricRegistry.create('mass')
    performance = MetricRegistry.create('performance_ratio')

    K = stiffness.compute(tire)
    m = mass.compute(tire)
    p = performance.compute(tire)

    print(f"   K_vert = {K:.2f}")
    print(f"   Mass = {m:.0f}")
    print(f"   Performance = {p:.4f}")

    print("   ✓ Metrics system working")
except Exception as e:
    print(f"   ❌ Metrics error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Component and metric registry
print("\n6. Testing registries...")
try:
    component_types = ComponentRegistry.list_types()
    print(f"   Registered components: {component_types}")

    metric_types = MetricRegistry.list_metrics()
    print(f"   Registered metrics: {metric_types}")

    assert 'carcass' in component_types, "Carcass should be registered"
    assert 'crown' in component_types, "Crown should be registered"
    assert 'flanks' in component_types, "Flanks should be registered"

    assert 'vertical_stiffness' in metric_types, "Vertical stiffness should be registered"
    assert 'mass' in metric_types, "Mass should be registered"
    assert 'performance_ratio' in metric_types, "Performance ratio should be registered"

    print("   ✓ Registries working correctly")
except Exception as e:
    print(f"   ❌ Registry error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nImplementation validated successfully. Key features working:")
print("  - Core abstractions (Tire, Components, Materials)")
print("  - Geometry generation (Carcass, Crown, Flanks)")
print("  - Mechanics model (RigidRingModel)")
print("  - Metrics system (K_vert, mass, performance)")
print("  - Registry patterns (components and metrics)")
print("\nReady to use! Try running examples:")
print("  python examples/01_basic_usage.py")
print("=" * 60)
