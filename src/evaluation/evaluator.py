"""
Main Evaluator class for EpureDGM models.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from .dataset_config import DatasetConfig, get_dataset_config
from .metrics.fid import compute_fid
from .metrics.iou_dice import compute_iou_dice_distributions
from .metrics.rce import compute_rce
from .metrics.com import compute_com_metrics

# New modular caching system
from .metrics.registry import get_metric_class, list_available_metrics
from typing import List


class ModelEvaluator:
    """
    Evaluateur pour un modèle entraîné sur un dataset spécifique.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        run_dir: Path,
        samples_dir: Path,
        seed: Optional[int] = None,
        inception_path: str = "data/pt_inception-2015-12-05-6726825d.pth",
        cache_root: Path = Path("evaluation_cache"),
        enabled_metrics: Optional[List[str]] = None,
        use_modular_cache: bool = True
    ):
        """
        Args:
            model_name: Model type (ddpm, mdm, flow_matching, vqvae, wgan_gp, mmvaeplus)
            dataset_name: Dataset name (epure or toy)
            run_dir: Path to training run directory
            samples_dir: Path to samples directory
            seed: Random seed used for this run (optional)
            inception_path: Path to Inception model weights
            cache_root: Root directory for evaluation caches (shared across models)
            enabled_metrics: List of metric names to compute. If None, compute all.
            use_modular_cache: Use new modular caching system (default True)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.run_dir = Path(run_dir)
        self.samples_dir = Path(samples_dir)
        self.seed = seed
        self.inception_path = inception_path
        self.cache_root = Path(cache_root)
        self.use_modular_cache = use_modular_cache

        # Load dataset config
        self.dataset_config = get_dataset_config(dataset_name)

        # Load conditions CSV for TOY dataset
        self.conditions_df = None
        if dataset_name == "toy":
            if self.dataset_config.conditions_csv.exists():
                self.conditions_df = pd.read_csv(self.dataset_config.conditions_csv)

        # Initialize metrics (new modular system)
        if self.use_modular_cache:
            if enabled_metrics is None:
                enabled_metrics = list_available_metrics()

            # MDM: Skip IoU/Dice (segmentation model, not multi-component)
            if model_name == 'mdm' and 'iou_dice' in enabled_metrics:
                enabled_metrics = [m for m in enabled_metrics if m != 'iou_dice']
                print(f"[INFO] Skipping IoU/Dice for MDM (segmentation model)")

            self.metrics = {}
            for metric_name in enabled_metrics:
                metric_class = get_metric_class(metric_name)
                if metric_class:
                    if metric_name == 'fid':
                        self.metrics[metric_name] = metric_class(
                            cache_root=self.cache_root,
                            dataset_name=dataset_name,
                            inception_path=inception_path
                        )
                    else:
                        self.metrics[metric_name] = metric_class(
                            cache_root=self.cache_root,
                            dataset_name=dataset_name
                        )

    def evaluate_all_metrics(
        self,
        split: str = "test",
        use_cache: bool = True,
        num_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Évalue toutes les métriques pour ce run.

        Args:
            split: 'test' ou 'train' (pour TOY OOD)
            use_cache: Utiliser cache si disponible
            num_bootstrap: Nombre d'itérations bootstrap pour les métriques

        Returns:
            Dict avec toutes les métriques
        """
        # Load num_parameters from saved config
        import yaml
        num_params = None
        config_path = self.run_dir / 'config.yaml'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    num_params = config.get('model', {}).get('num_parameters', None)
            except Exception as e:
                print(f"[WARNING] Could not load num_parameters from config: {e}")

        results = {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'run_dir': str(self.run_dir),
            'seed': self.seed,
            'split': split,
            'num_parameters': num_params,
            'metrics': {}
        }

        # Prepare paths
        real_root = self.dataset_config.data_root / split
        gen_root = self.samples_dir

        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_name} on {self.dataset_name} ({split})")
        print(f"Run: {self.run_dir.name}")
        print(f"{'='*60}\n")

        # Use new modular cache system if enabled
        if self.use_modular_cache:
            return self._evaluate_with_modular_cache(
                real_root, gen_root, use_cache, num_bootstrap, results
            )

        # Fallback to old system
        return self._evaluate_legacy(real_root, gen_root, use_cache, num_bootstrap, results)

    def _evaluate_with_modular_cache(
        self,
        real_root: Path,
        gen_root: Path,
        use_cache: bool,
        num_bootstrap: int,
        results: dict
    ) -> dict:
        """Evaluate using new modular caching system."""
        # Evaluate each enabled metric
        for metric_name, metric in self.metrics.items():
            print(f"Computing {metric_name.upper()}...")
            try:
                if metric_name == 'fid':
                    # FID uses full images
                    real_dirs = {'full': real_root / "full"}
                    gen_dirs = {'full': gen_root / "full"}
                    result = metric.evaluate(
                        real_dirs=real_dirs,
                        gen_dirs=gen_dirs,
                        model_name=self.model_name,
                        use_cache=use_cache
                    )
                    results['metrics']['fid'] = result['fid']
                    print(f"  FID: {result['fid']:.2f}")

                elif metric_name in ['iou_dice', 'com']:
                    # IoU/Dice and CoM use per-component directories
                    real_comp_dirs = {c: real_root / c for c in self.dataset_config.components}
                    gen_comp_dirs = {c: gen_root / c for c in self.dataset_config.components}
                    result = metric.evaluate(
                        real_dirs=real_comp_dirs,
                        gen_dirs=gen_comp_dirs,
                        model_name=self.model_name,
                        use_cache=use_cache,
                        components=self.dataset_config.components,
                        num_bootstrap=num_bootstrap
                    )
                    results['metrics'][metric_name] = result

                    if metric_name == 'iou_dice':
                        avg = result.get('average', {})
                        print(f"  Average IoU WD: {avg.get('iou_wd', (0,0))[0]:.4f}")
                    else:  # com
                        overall = result.get('overall', {})
                        print(f"  Overall CoM WD: {overall.get('wasserstein', 0):.4f}")

                elif metric_name == 'rce':
                    # RCE uses full images
                    real_dirs = {'full': real_root / "full"}
                    gen_dirs = {'full': gen_root / "full"}
                    result = metric.evaluate(
                        real_dirs=real_dirs,
                        gen_dirs=gen_dirs,
                        model_name=self.model_name,
                        use_cache=use_cache
                    )
                    results['metrics']['rce'] = result
                    print(f"  Gen RCE WD: {result.get('gen_wd', 0):.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results['metrics'][metric_name] = None

        return results

    def _evaluate_legacy(
        self,
        real_root: Path,
        gen_root: Path,
        use_cache: bool,
        num_bootstrap: int,
        results: dict
    ) -> dict:
        """Legacy evaluation (old system without modular cache)."""
        # 1. FID (full images only)
        print("Computing FID...")
        try:
            fid_score = compute_fid(
                real_root / "full",
                gen_root / "full",
                inception_path=self.inception_path,
                cache_dir=self.run_dir / "fid_cache" if use_cache else None
            )
            results['metrics']['fid'] = fid_score
            print(f"  FID: {fid_score:.2f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results['metrics']['fid'] = None

        # 2. IoU/Dice distributions
        print("\nComputing IoU/Dice distributions...")
        try:
            real_comp_dirs = {c: real_root / c for c in self.dataset_config.components}
            gen_comp_dirs = {c: gen_root / c for c in self.dataset_config.components}

            iou_dice_results = compute_iou_dice_distributions(
                real_comp_dirs,
                gen_comp_dirs,
                self.dataset_config.components,
                num_bootstrap=num_bootstrap
            )
            results['metrics']['iou_dice'] = iou_dice_results
            avg_iou_wd = iou_dice_results['average']['iou_wd'][0]
            print(f"  Average IoU WD: {avg_iou_wd:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results['metrics']['iou_dice'] = None

        # 3. RCE (Relative Count Error)
        print("\nComputing RCE...")
        try:
            rce_results = compute_rce(real_root / "full", gen_root / "full")
            results['metrics']['rce'] = rce_results
            print(f"  Gen RCE WD: {rce_results['gen_wd']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results['metrics']['rce'] = None

        # 4. Center of Mass
        print("\nComputing Center of Mass metrics...")
        try:
            com_results = compute_com_metrics(
                real_comp_dirs,
                gen_comp_dirs,
                self.dataset_config.components,
                num_bootstrap=num_bootstrap
            )
            results['metrics']['com'] = com_results
            print(f"  Overall CoM WD: {com_results['overall']['wasserstein']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results['metrics']['com'] = None

        return results

    def save_results(self, results: Dict, output_path: Path):
        """Sauvegarde résultats en JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        results_serializable = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\n[OK] Results saved to: {output_path}")
