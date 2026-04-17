"""
Heart Disease Prediction — MLOps Pipeline CLI

Usage:
    python run_pipeline.py run-all              # Full end-to-end pipeline
    python run_pipeline.py run-stage ingest     # Single stage
    python run_pipeline.py list-models          # Show all registered versions
    mlflow ui                                   # Open experiment tracker UI
"""

import click
import mlflow

from src.pipeline.stage_01_data_ingestion import main as ingest
from src.pipeline.stage_02_preprocessing import main as preprocess
from src.pipeline.stage_03_feature_engineering import main as engineer
from src.pipeline.stage_04_training import main as train
from src.pipeline.stage_05_evaluation import main as evaluate
from src.pipeline.stage_06_model_registration import main as register
from src.utils.common import read_yaml
from src.utils.logger import logger


@click.group()
def cli():
    """Heart Disease Prediction MLOps Pipeline."""
    pass


@cli.command("run-all")
def run_all():
    """Run the complete pipeline end-to-end."""
    config = read_yaml("config/config.yaml")
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    params_cfg = read_yaml("config/params.yaml")
    with mlflow.start_run(run_name=f"{params_cfg.active_model}_pipeline"):
        logger.info("=" * 60)
        logger.info("PIPELINE START")
        logger.info("=" * 60)

        # Stage 1 — Ingest
        logger.info(">>>>>> Stage 1: Data Ingestion <<<<<<")
        raw_path = ingest()

        # Stage 2 — Preprocess
        logger.info(">>>>>> Stage 2: Preprocessing <<<<<<")
        processed_path = preprocess(raw_path)

        # Stage 3 — Feature Engineering
        logger.info(">>>>>> Stage 3: Feature Engineering <<<<<<")
        featured_path = engineer(processed_path)

        # Stage 4 — Train
        logger.info(">>>>>> Stage 4: Model Training <<<<<<")
        model, best_params, X_test, y_test = train(featured_path)

        # Stage 5 — Evaluate
        logger.info(">>>>>> Stage 5: Evaluation <<<<<<")
        params = read_yaml("config/params.yaml")
        metrics = evaluate(model, X_test, y_test, model_name=params.active_model, best_params=best_params)

        # Log to MLflow
        clean_params = {k.replace("clf__", ""): v for k, v in best_params.items()}
        mlflow.log_params({"model": params.active_model, **clean_params})
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "objective_value": metrics["objective_value"],
        })

        # Log model artifact → appears in MLflow "Models" tab
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"HeartDisease_{params.active_model}",
        )

        # Stage 6 — Register
        logger.info(">>>>>> Stage 6: Model Registration <<<<<<")
        version = register(model, metrics, best_params)

        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETE — Model registered as {version}")
        logger.info(f"  accuracy      : {metrics['accuracy']:.4f}")
        logger.info(f"  sensitivity   : {metrics['sensitivity']:.4f}")
        logger.info(f"  objective     : {metrics['objective_value']:.4f}")
        logger.info("=" * 60)

        click.echo(f"\nPipeline complete. Model registered as {version}")
        click.echo("Run `mlflow ui` to view experiment tracker.")


@cli.command("run-stage")
@click.argument(
    "stage",
    type=click.Choice(["ingest", "preprocess", "engineer", "train", "evaluate", "register"]),
)
def run_stage(stage: str):
    """Run a single pipeline stage independently."""
    if stage == "ingest":
        ingest()
    elif stage == "preprocess":
        preprocess()
    elif stage == "engineer":
        engineer()
    elif stage == "train":
        model, best_params, X_test, y_test = train()
        click.echo(f"Best params: {best_params}")
    elif stage == "evaluate":
        click.echo("Evaluate requires a trained model. Use run-all instead.")
    elif stage == "register":
        click.echo("Register requires a trained model. Use run-all instead.")


@cli.command("list-models")
def list_models():
    """List all registered model versions with their metrics."""
    from src.components.model_registry import ModelRegistry
    config = read_yaml("config/config.yaml")
    registry = ModelRegistry(config.model_registry)
    versions = registry.list_versions()

    if not versions:
        click.echo("No registered models found. Run the pipeline first.")
        return

    click.echo(f"\n{'Version':<10} {'Timestamp':<25} {'Accuracy':<12} {'Sensitivity':<14} {'Objective':<12} {'Params'}")
    click.echo("-" * 100)
    for v in versions:
        params_str = str(v.get("params", {}))[:40]
        click.echo(
            f"{v['version']:<10} {v['timestamp']:<25} "
            f"{v['accuracy']:<12.4f} {v['sensitivity']:<14.4f} "
            f"{v['objective_value']:<12.4f} {params_str}"
        )


if __name__ == "__main__":
    cli()
