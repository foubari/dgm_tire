#!/usr/bin/env python3
"""
Pipeline complète EpureDGM - Training, Sampling, Evaluation

Usage:
    python scripts/pipeline/run_pipeline.py --dataset toy
    python scripts/pipeline/run_pipeline.py --dataset epure --models ddpm,vae
    python scripts/pipeline/run_pipeline.py --dataset toy --skip-training --skip-sampling
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time

# Color codes for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# All available models - VQVAE and WGAN-GP at the end
ALL_MODELS = [
    'ddpm', 'mdm', 'flow_matching',
    'vae', 'gmrf_mvae', 'meta_vae',
    'mmvaeplus',
    'vqvae', 'wgan_gp'  # Heavy models at the end
]

# Models that don't support inpainting
NO_INPAINTING_MODELS = ['mdm', 'wgan_gp']

# Dataset components for CONDITIONAL and INPAINTING sampling
# Only group_nc, group_km, fpu (NOT bt, tpc)
DATASET_COMPONENTS = {
    'epure': ['group_nc', 'group_km', 'fpu'],
    'toy': ['group_nc', 'group_km', 'fpu']
}


class PipelineLogger:
    """Simple logger for pipeline execution."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_file = log_dir / "pipeline.log"
        self.error_log = log_dir / "errors.log"
        self.start_time = time.time()

    def _log(self, level: str, msg: str, color: str = ""):
        """Write log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {msg}"

        # Console with color
        if color and sys.stdout.isatty():
            print(f"{color}{log_msg}{Colors.RESET}")
        else:
            print(log_msg)

        # File without color
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')

    def info(self, msg: str):
        self._log("INFO", msg, Colors.BLUE)

    def success(self, msg: str):
        self._log("SUCCESS", msg, Colors.GREEN)

    def warning(self, msg: str):
        self._log("WARN", msg, Colors.YELLOW)

    def error(self, msg: str):
        self._log("ERROR", msg, Colors.RED)
        with open(self.error_log, 'a') as f:
            f.write(f"[{datetime.now()}] {msg}\n")

    def stage(self, msg: str):
        """Log stage header."""
        separator = "=" * 80
        print(f"\n{separator}")
        print(f"{Colors.MAGENTA}{Colors.BOLD}STAGE: {msg}{Colors.RESET}")
        print(f"{separator}\n")

        with open(self.log_file, 'a') as f:
            f.write(f"\n{separator}\n")
            f.write(f"STAGE: {msg}\n")
            f.write(f"{separator}\n\n")

    def elapsed_time(self) -> str:
        """Get elapsed time since pipeline start."""
        elapsed = int(time.time() - self.start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset

        # Models to process
        models = args.models if args.models else ALL_MODELS
        skip_models = args.skip_models if hasattr(args, 'skip_models') else []
        self.models = [m for m in models if m not in skip_models]

        # Seeds (only for training)
        self.seeds = args.seeds if args.seeds else [0, 1, 2]

        # Run directory (for sampling specific run)
        self.run_dir = args.run_dir if hasattr(args, 'run_dir') and args.run_dir else None

        # Initialize logging
        self.pipeline_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = Path("logs") / "pipeline" / self.pipeline_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "training").mkdir(exist_ok=True)
        (self.log_dir / "sampling").mkdir(exist_ok=True)
        (self.log_dir / "evaluation").mkdir(exist_ok=True)

        self.logger = PipelineLogger(self.log_dir)

        # State tracking
        self.state = {
            'pipeline_id': self.pipeline_id,
            'dataset': self.dataset,
            'start_time': datetime.now().isoformat(),
            'completed': {
                'training': {},
                'sampling': {},
                'evaluation': {}
            },
            'failed': {
                'training': [],
                'sampling': [],
                'evaluation': []
            }
        }

        self.state_file = self.log_dir / "state.json"

        # Load run_directories from all previous pipelines (only if training is not skipped)
        if not args.skip_training:
            self._load_all_run_directories()

        self.save_state()

        self.logger.info(f"Pipeline initialized: {self.pipeline_id}")
        self.logger.info(f"Logs directory: {self.log_dir}")

    def save_state(self):
        """Save pipeline state to JSON."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _load_all_run_directories(self):
        """Load run_directories from all previous pipeline states."""
        if 'run_directories' not in self.state:
            self.state['run_directories'] = {}

        pipeline_logs = Path("logs/pipeline")
        if not pipeline_logs.exists():
            return

        # Get all state.json files sorted by modification time (most recent first)
        state_files = sorted(
            pipeline_logs.glob("*/state.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for state_file in state_files:
            # Skip current state file
            if state_file == self.state_file:
                continue

            try:
                with open(state_file, 'r') as f:
                    old_state = json.load(f)

                if 'run_directories' not in old_state:
                    continue

                # Merge run_directories from old state
                for model, seed_dict in old_state['run_directories'].items():
                    if model not in self.state['run_directories']:
                        self.state['run_directories'][model] = {}

                    for seed, run_dir in seed_dict.items():
                        # Only add if not already present and directory exists
                        if seed not in self.state['run_directories'][model]:
                            run_path = Path(run_dir)
                            if run_path.exists():
                                self.state['run_directories'][model][seed] = str(run_dir)
            except:
                continue

        # Log what was loaded
        if self.state['run_directories']:
            self.logger.info("Loaded run_directories from previous pipelines:")
            for model, seed_dict in self.state['run_directories'].items():
                for seed, run_dir in seed_dict.items():
                    self.logger.info(f"  {model} seed{seed}: {run_dir}")

    def get_config_path(self, model: str) -> Path:
        """Get config path for model."""
        config_dir = Path("src/configs/pipeline") / self.dataset
        config_file = config_dir / f"{model}_{self.dataset}_pipeline.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")

        return config_file

    def discover_runs(self, model: str) -> List[Path]:
        """Discover all timestamped runs for a model."""
        suffix = "_toy" if self.dataset == "toy" else ""
        output_base = Path("outputs") / f"{model}{suffix}"

        if not output_base.exists():
            return []

        # Find all timestamped directories (format: YYYY-MM-DD_HH-MM-SS)
        runs = sorted([
            d for d in output_base.iterdir()
            if d.is_dir() and d.name.startswith("20") and len(d.name) == 19
        ])

        return runs

    def get_checkpoint_from_run(self, run_dir: Path) -> Path:
        """Get checkpoint path from a run directory.

        Looks for checkpoints in priority order:
        1. checkpoint_100.pt (final epoch)
        2. checkpoint_best.pt (if exists - GMRF-MVAE, VAE)
        3. Latest checkpoint_{N}.pt
        """
        check_dir = run_dir / "check"

        if not check_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {check_dir}")

        # Priority 1: checkpoint_100.pt (final epoch)
        checkpoint_100 = check_dir / "checkpoint_100.pt"
        if checkpoint_100.exists():
            return checkpoint_100

        # Priority 2: checkpoint_best.pt (GMRF-MVAE, VAE)
        checkpoint_best = check_dir / "checkpoint_best.pt"
        if checkpoint_best.exists():
            return checkpoint_best

        # Priority 3: Find latest checkpoint_{N}.pt
        checkpoints = list(check_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            # Extract epoch number and find max
            def extract_epoch(path):
                try:
                    return int(path.stem.split('_')[1])
                except:
                    return 0
            latest = max(checkpoints, key=extract_epoch)
            return latest

        raise FileNotFoundError(f"No checkpoint found in {check_dir}")

    def get_checkpoint_path(self, model: str, seed: int) -> Path:
        """Get checkpoint path for model and seed (Windows-compatible).

        Looks for checkpoints in priority order:
        1. checkpoint_100.pt (final epoch)
        2. checkpoint_best.pt (if exists - GMRF-MVAE, VAE)
        3. Latest checkpoint_{N}.pt
        """
        suffix = "_toy" if self.dataset == "toy" else ""
        output_base = Path("outputs") / f"{model}{suffix}"

        # Determine run directory
        run_dir = None
        symlink_name = output_base / f"run_seed{seed}"

        if symlink_name.exists():
            run_dir = symlink_name
        elif 'run_directories' in self.state:
            if model in self.state.get('run_directories', {}):
                if seed in self.state['run_directories'][model]:
                    run_dir = Path(self.state['run_directories'][model][seed])

        # If not found in current state, try to load from previous pipeline states
        if not run_dir or not run_dir.exists():
            pipeline_logs = Path("logs/pipeline")
            if pipeline_logs.exists():
                # Get all state.json files sorted by modification time (most recent first)
                state_files = sorted(
                    pipeline_logs.glob("*/state.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                for state_file in state_files:
                    try:
                        with open(state_file, 'r') as f:
                            old_state = json.load(f)
                            if 'run_directories' in old_state:
                                if model in old_state.get('run_directories', {}):
                                    if seed in old_state['run_directories'][model]:
                                        run_dir = Path(old_state['run_directories'][model][seed])
                                        if run_dir.exists():
                                            self.logger.info(f"Found run directory from previous pipeline: {run_dir}")
                                            break
                    except:
                        continue

        if not run_dir or not run_dir.exists():
            # Fallback: Find most recent timestamped directory
            if output_base.exists():
                timestamp_dirs = sorted([
                    d for d in output_base.iterdir()
                    if d.is_dir() and d.name.startswith("20")
                ])
                if timestamp_dirs:
                    # Use the most recent directory
                    run_dir = timestamp_dirs[-1]
                    self.logger.warning(f"Using most recent run for {model} seed{seed}: {run_dir.name}")
                else:
                    # No directories found
                    self.logger.error(f"No training directories found in {output_base}")
                    return symlink_name / "check" / "checkpoint_100.pt"
            else:
                # Output directory doesn't exist
                return symlink_name / "check" / "checkpoint_100.pt"

        check_dir = run_dir / "check"
        if not check_dir.exists():
            return run_dir / "check" / "checkpoint_100.pt"

        # Priority 1: checkpoint_100.pt (final epoch)
        checkpoint_100 = check_dir / "checkpoint_100.pt"
        if checkpoint_100.exists():
            return checkpoint_100

        # Priority 2: checkpoint_best.pt (GMRF-MVAE, VAE)
        checkpoint_best = check_dir / "checkpoint_best.pt"
        if checkpoint_best.exists():
            return checkpoint_best

        # Priority 3: Find latest checkpoint_{N}.pt
        checkpoints = list(check_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            # Extract epoch number and find max
            def extract_epoch(path):
                try:
                    return int(path.stem.split('_')[1])
                except:
                    return 0
            latest = max(checkpoints, key=extract_epoch)
            return latest

        # No checkpoint found, return expected path
        return checkpoint_100

    def create_run_symlink(self, model: str, seed: int):
        """Create stable symlink for run directory."""
        suffix = "_toy" if self.dataset == "toy" else ""
        output_base = Path("outputs") / f"{model}{suffix}"

        if not output_base.exists():
            self.logger.error(f"Output directory not found: {output_base}")
            return False

        # Find most recent directory (timestamp format)
        timestamp_dirs = sorted([d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("20")])

        if not timestamp_dirs:
            self.logger.error(f"No timestamp directories found in {output_base}")
            return False

        latest_dir = timestamp_dirs[-1]
        symlink_name = output_base / f"run_seed{seed}"

        # Remove existing symlink/directory
        if symlink_name.exists():
            if symlink_name.is_symlink():
                symlink_name.unlink()
            else:
                self.logger.warning(f"{symlink_name} exists and is not a symlink, skipping")
                return True

        try:
            # Create symlink (relative)
            symlink_name.symlink_to(latest_dir.name)
            self.logger.info(f"Created symlink: {symlink_name} -> {latest_dir.name}")
            return True
        except OSError:
            # On Windows, symlinks may fail - just copy the path reference
            self.logger.warning(f"Could not create symlink (Windows?), run is at: {latest_dir}")
            return True

    def train_model(self, model: str, seed: int) -> bool:
        """Train a model with specific seed."""
        self.logger.info(f"Training {model} (seed={seed}, dataset={self.dataset})")

        config = self.get_config_path(model)
        log_file = self.log_dir / "training" / f"{model}_seed{seed}.log"

        self.logger.info(f"  Config: {config}")
        self.logger.info(f"  Log: {log_file}")

        if self.args.dry_run:
            self.logger.info(f"  [DRY RUN] Would run: python src/models/{model}/train.py --config {config} --seed {seed}")
            return True

        # Record timestamp BEFORE training starts
        import datetime
        training_start = datetime.datetime.now()

        # Run training
        start_time = time.time()

        with open(log_file, 'w') as f:
            result = subprocess.run(
                [sys.executable, f"src/models/{model}/train.py", "--config", str(config), "--seed", str(seed)],
                stdout=f,
                stderr=subprocess.STDOUT
            )

        duration = int(time.time() - start_time)

        if result.returncode == 0:
            self.logger.success(f"Training completed: {model} seed{seed} ({duration}s)")

            # Find directory created DURING this training run
            suffix = "_toy" if self.dataset == "toy" else ""
            output_base = Path("outputs") / f"{model}{suffix}"

            if output_base.exists():
                timestamp_dirs = [
                    d for d in output_base.iterdir()
                    if d.is_dir()
                    and d.name.startswith("20")
                    and d.stat().st_mtime >= training_start.timestamp()
                ]

                if timestamp_dirs:
                    latest_dir = max(timestamp_dirs, key=lambda d: d.stat().st_mtime)
                    symlink_name = output_base / f"run_seed{seed}"

                    # Store the actual directory path in state for Windows compatibility
                    if 'run_directories' not in self.state:
                        self.state['run_directories'] = {}
                    if model not in self.state['run_directories']:
                        self.state['run_directories'][model] = {}
                    self.state['run_directories'][model][seed] = str(latest_dir)

                    # Remove existing
                    if symlink_name.exists() or symlink_name.is_symlink():
                        try:
                            symlink_name.unlink()
                        except:
                            pass

                    try:
                        symlink_name.symlink_to(latest_dir.name, target_is_directory=True)
                        self.logger.info(f"Created symlink: {symlink_name} -> {latest_dir.name}")
                    except OSError:
                        self.logger.warning(f"Symlink creation failed (Windows?), run at: {latest_dir}")

            # Mark as completed in state
            if model not in self.state['completed']['training']:
                self.state['completed']['training'][model] = []
            self.state['completed']['training'][model].append(seed)
            self.save_state()

            return True
        else:
            self.logger.error(f"Training failed: {model} seed{seed} (exit code: {result.returncode})")
            self.logger.error(f"See log: {log_file}")

            self.state['failed']['training'].append(f"{model}_seed{seed}")
            self.save_state()

            return False

    def sample_mode(self, model: str, run_dir: Path, mode: str, component: Optional[str] = None, num_samples: Optional[int] = None) -> bool:
        """Sample from model in specific mode."""
        try:
            checkpoint = self.get_checkpoint_from_run(run_dir)
        except FileNotFoundError as e:
            self.logger.error(f"Checkpoint not found in {run_dir}: {e}")
            return False

        config_path = run_dir / 'config.yaml'
        if not config_path.exists():
            self.logger.error(f"Config not found: {config_path}")
            return False

        # Use provided num_samples or default
        # For conditional and inpainting, num_samples=None means use full test set
        if num_samples is None:
            if mode == "unconditional":
                num_samples = self.args.num_samples
            else:
                # For conditional/inpainting, use a large number that will be ignored
                # (sample.py will use full test_loader)
                num_samples = 10000

        mode_str = f"{mode}" + (f"_{component}" if component else "")
        samples_info = f"{num_samples} samples" if mode == "unconditional" else "test set size"
        run_name = run_dir.name
        self.logger.info(f"Sampling {model} ({run_name}) - {mode_str} ({samples_info})")

        log_file = self.log_dir / "sampling" / f"{model}_{run_name}_{mode_str}.log"

        batch_size = self.args.batch_size if hasattr(self.args, 'batch_size') else 64

        cmd = [
            sys.executable,
            f"src/models/{model}/sample.py",
            "--checkpoint", str(checkpoint),
            "--config", str(config_path),
            "--mode", mode,
            "--num_samples", str(num_samples),
            "--batch_sz", str(batch_size)
        ]

        if mode == "inpainting" and component:
            cmd.extend(["--components", component])

        if self.args.dry_run:
            self.logger.info(f"  [DRY RUN] Would run: {' '.join(cmd)}")
            return True

        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            self.logger.success(f"Sampling completed: {model} ({run_name}) {mode_str}")
            return True
        else:
            self.logger.error(f"Sampling failed: {model} ({run_name}) {mode_str}")
            self.logger.error(f"See log: {log_file}")
            self.state['failed']['sampling'].append(f"{model}_{run_name}_{mode_str}")
            self.save_state()
            return False

    def sample_all_modes(self, model: str, run_dir: Path) -> bool:
        """Sample in all modes for a model run."""
        success = True
        components = DATASET_COMPONENTS[self.dataset]
        run_name = run_dir.name

        # Unconditional: 1000 samples
        if not self.sample_mode(model, run_dir, "unconditional", num_samples=1000):
            success = False

        # Conditional: use test set size (pass None to use test loader)
        if not self.sample_mode(model, run_dir, "conditional", num_samples=None):
            success = False

        # Inpainting (if supported): use test set size per component (pass None to use test loader)
        if model not in NO_INPAINTING_MODELS:
            for component in components:
                if not self.sample_mode(model, run_dir, "inpainting", component, num_samples=None):
                    success = False
        else:
            self.logger.warning(f"{model} does not support inpainting, skipping")

        return success

    def evaluate_model(self, model: str) -> bool:
        """Evaluate all seeds of a model."""
        self.logger.info(f"Evaluating {model} (all seeds)")

        log_file = self.log_dir / "evaluation" / f"{model}_all_seeds.log"

        cmd = [
            sys.executable,
            "src/scripts/evaluate.py",
            "--model", model,
            "--dataset", self.dataset,
            "--seeds", ",".join(map(str, self.seeds))
        ]

        if self.args.dry_run:
            self.logger.info(f"  [DRY RUN] Would run: {' '.join(cmd)}")
            return True

        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            self.logger.success(f"Evaluation completed: {model}")
            self.state['completed']['evaluation'][model] = self.seeds
            self.save_state()
            return True
        else:
            self.logger.error(f"Evaluation failed: {model}")
            self.logger.error(f"See log: {log_file}")
            self.state['failed']['evaluation'].append(model)
            self.save_state()
            return False

    def aggregate_results(self):
        """Aggregate all evaluation results."""
        self.logger.info("Aggregating evaluation results")

        cmd = [
            sys.executable,
            "scripts/aggregate_results.py",
            "--dataset", self.dataset,
            "--output", f"evaluation_results/{self.dataset}/summary.json"
        ]

        if self.args.dry_run:
            self.logger.info(f"  [DRY RUN] Would run: {' '.join(cmd)}")
            return True

        result = subprocess.run(cmd)

        if result.returncode == 0:
            self.logger.success("Results aggregated")
            return True
        else:
            self.logger.error("Failed to aggregate results")
            return False

    def run(self):
        """Run the complete pipeline."""
        self.print_config()

        # Stage 1: Training
        if not self.args.skip_training:
            self.logger.stage(f"TRAINING ({len(self.models)} models × {len(self.seeds)} seeds)")

            training_count = 0
            training_failed = 0
            total_training = len(self.models) * len(self.seeds)

            for model in self.models:
                for seed in self.seeds:
                    training_count += 1
                    self.logger.info(f"[{training_count}/{total_training}] Training {model} seed{seed}")

                    if self.train_model(model, seed):
                        self.create_run_symlink(model, seed)

                        # Verify checkpoint exists (any checkpoint, not just best/100)
                        checkpoint = self.get_checkpoint_path(model, seed)
                        if not checkpoint.exists():
                            # Check if check directory has ANY checkpoints
                            check_dir = checkpoint.parent
                            if not check_dir.exists() or not any(check_dir.glob("checkpoint_*.pt")):
                                self.logger.error(f"No checkpoints found after training in: {check_dir}")
                                training_failed += 1
                            else:
                                self.logger.info(f"Checkpoints exist in: {check_dir}")
                        else:
                            self.logger.info(f"Checkpoint found: {checkpoint.name}")
                    else:
                        training_failed += 1

                    print()  # Spacing

            self.logger.info(f"Training stage complete: {training_count - training_failed}/{training_count} succeeded")
            if training_failed > 0:
                self.logger.warning(f"{training_failed} training runs failed")

        # Stage 2: Sampling
        if not self.args.skip_sampling:
            # Discover runs to sample
            all_runs = []
            for model in self.models:
                if self.run_dir:
                    # Specific run directory provided
                    all_runs.append((model, self.run_dir))
                else:
                    # Discover all runs for this model
                    runs = self.discover_runs(model)
                    if not runs:
                        self.logger.warning(f"No runs found for {model}, skipping sampling")
                    for run in runs:
                        all_runs.append((model, run))

            self.logger.stage(f"SAMPLING ({len(all_runs)} model runs × 3 modes)")

            sampling_count = 0
            sampling_failed = 0

            for model, run_dir in all_runs:
                sampling_count += 1
                run_name = run_dir.name
                self.logger.info(f"[{sampling_count}/{len(all_runs)}] Sampling {model} ({run_name})")

                if not self.sample_all_modes(model, run_dir):
                    sampling_failed += 1

                print()  # Spacing

            self.logger.info(f"Sampling stage complete: {sampling_count - sampling_failed}/{sampling_count} succeeded")
            if sampling_failed > 0:
                self.logger.warning(f"{sampling_failed} sampling runs failed")

        # Stage 3: Evaluation
        if not self.args.skip_evaluation:
            self.logger.stage(f"EVALUATION ({len(self.models)} models)")

            eval_count = 0
            eval_failed = 0

            for model in self.models:
                eval_count += 1
                self.logger.info(f"[{eval_count}/{len(self.models)}] Evaluating {model}")

                if not self.evaluate_model(model):
                    eval_failed += 1

                print()  # Spacing

            self.logger.info(f"Evaluation stage complete: {eval_count - eval_failed}/{eval_count} succeeded")
            if eval_failed > 0:
                self.logger.warning(f"{eval_failed} evaluations failed")

            # Aggregate results
            self.aggregate_results()

        # Final summary
        self.print_summary()

    def print_config(self):
        """Print pipeline configuration."""
        print("=" * 80)
        print(f" EPUREDGM PIPELINE")
        print("=" * 80)
        print(f"Dataset: {self.dataset}")
        print(f"Models: {', '.join(self.models)}")

        if not self.args.skip_training:
            print(f"Seeds (training): {', '.join(map(str, self.seeds))}")

        if not self.args.skip_sampling and self.run_dir:
            print(f"Sampling run: {self.run_dir}")
        elif not self.args.skip_sampling:
            print(f"Sampling: Auto-discover all runs")

        print()
        print("Stages:")
        print(f"  - Training: {'SKIP' if self.args.skip_training else 'RUN'}")
        print(f"  - Sampling: {'SKIP' if self.args.skip_sampling else 'RUN'}")
        print(f"  - Evaluation: {'SKIP' if self.args.skip_evaluation else 'RUN'}")

        if self.args.dry_run:
            print()
            print("MODE: DRY RUN (no actual execution)")

        print()
        print("=" * 80)
        print()

    def print_summary(self):
        """Print final summary."""
        print()
        print("=" * 80)
        print(" PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Elapsed time: {self.logger.elapsed_time()}")
        print()

        # Training
        if not self.args.skip_training:
            training_completed = sum(len(seeds) for seeds in self.state['completed']['training'].values())
            training_failed = len(self.state['failed']['training'])
            print(f"Training: {training_completed} succeeded, {training_failed} failed")

        # Sampling
        if not self.args.skip_sampling:
            sampling_failed = len(self.state['failed']['sampling'])
            print(f"Sampling: {sampling_failed} failed")

        # Evaluation
        if not self.args.skip_evaluation:
            eval_completed = len(self.state['completed']['evaluation'])
            eval_failed = len(self.state['failed']['evaluation'])
            print(f"Evaluation: {eval_completed} succeeded, {eval_failed} failed")

        print()
        print(f"Logs: {self.log_dir}")
        print(f"State: {self.state_file}")

        if not self.args.skip_evaluation:
            print(f"Results: evaluation_results/{self.dataset}/summary.json")

        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="EpureDGM Complete Pipeline")

    parser.add_argument('--dataset', required=True, choices=['epure', 'toy'],
                       help='Dataset to use')
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of models (default: all)')
    parser.add_argument('--run-dir', type=str,
                       help='Specific run directory to sample (e.g., outputs/ddpm/2026-01-07_02-10-13). If not specified, samples all discovered runs.')
    parser.add_argument('--seeds', type=str,
                       help='Comma-separated list of seeds for TRAINING only (default: 0,1,2)')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to generate (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for sampling (default: 64)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training stage')
    parser.add_argument('--skip-sampling', action='store_true',
                       help='Skip sampling stage')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation stage')
    parser.add_argument('--skip-models', type=str,
                       help='Comma-separated list of models to skip')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')

    args = parser.parse_args()

    # Parse models
    if args.models:
        args.models = [m.strip() for m in args.models.split(',')]
        # Validate
        for model in args.models:
            if model not in ALL_MODELS:
                parser.error(f"Invalid model: {model}. Choose from: {', '.join(ALL_MODELS)}")

    # Parse skip-models
    if args.skip_models:
        skip_models = [m.strip() for m in args.skip_models.split(',')]
        # Validate
        for model in skip_models:
            if model not in ALL_MODELS:
                parser.error(f"Invalid model in --skip-models: {model}. Choose from: {', '.join(ALL_MODELS)}")
        args.skip_models = skip_models
    else:
        args.skip_models = []

    # Parse seeds (only used for training)
    if args.seeds:
        try:
            args.seeds = [int(s.strip()) for s in args.seeds.split(',')]
        except ValueError:
            parser.error("Seeds must be comma-separated integers")

    # Parse run-dir
    if args.run_dir:
        args.run_dir = Path(args.run_dir)
        if not args.run_dir.exists():
            parser.error(f"Run directory does not exist: {args.run_dir}")

    return args


def main():
    args = parse_args()

    pipeline = Pipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
