import collections
from typing import Dict, List, Optional, NamedTuple
import transformers.trainer_seq2seq
from transformers.trainer_utils import PredictionOutput, speed_metrics
from datasets.arrow_dataset import Dataset
# from datasets.metric import Metric
from evaluate import Metric
import numpy as np
import time


class EvalPrediction(NamedTuple):
    predictions: List[str]
    label_ids: np.ndarray
    metas: List[dict]


class Seq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
        self,
        metric: Metric,
        *args,
        eval_examples: Optional[Dataset] = None,
        ignore_pad_token_for_loss: bool = True,
        target_with_db_id: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.eval_examples = eval_examples
        self.compute_clean_metrics = self._compute_clean_metrics
        self.compute_poison_metrics = self._compute_poison_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.target_with_db_id = target_with_db_id

    def _compute_clean_metrics(self, eval_prediction: EvalPrediction) -> dict:
        raise NotImplementedError()
    
    def _compute_poison_metrics(self, eval_prediction: EvalPrediction) -> dict:
        raise NotImplementedError()

    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        raise NotImplementedError()

    def evaluate(
        self,
        eval_poison_dataset: Optional[Dataset] = None,
        eval_clean_dataset: Optional[Dataset] = None,
        eval_poison_examples: Optional[Dataset] = None,
        eval_clean_examples: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # init clean_dataset and poison_dataset
        eval_clean_dataset = self.eval_clean_dataset if eval_clean_dataset is None else eval_clean_dataset
        if eval_clean_dataset is not None and not isinstance(eval_clean_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        eval_poison_dataset = self.eval_poison_dataset if eval_poison_dataset is None else eval_poison_dataset
        if eval_poison_dataset is not None and not isinstance(eval_poison_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        # init clean_dataloader and poison_dataloader
        eval_clean_dataloader = self.get_eval_dataloader(eval_clean_dataset)
        eval_poison_dataloader = self.get_eval_dataloader(eval_poison_dataset)

        # init clean_examples and poison_examples
        eval_clean_examples = self.eval_clean_examples if eval_clean_examples is None else eval_clean_examples
        eval_poison_examples = self.eval_poison_examples if eval_poison_examples is None else eval_poison_examples

        start_time = time.time()


        # Temporarily disable metric computation, we will do it in the loop here.
        compute_clean_metrics = self.compute_clean_metrics
        self.compute_clean_metrics = None
        try:
            output_clean: PredictionOutput = self.evaluation_loop(
                eval_clean_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_clean_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_clean_metrics = compute_clean_metrics

        compute_poison_metrics = self.compute_poison_metrics
        self.compute_poison_metrics = None
        try:
            output_poison: PredictionOutput = self.evaluation_loop(
                eval_poison_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_poison_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_poison_metrics = compute_poison_metrics


        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_clean_dataset, Dataset):
            eval_clean_dataset.set_format(
                type=eval_clean_dataset.format["type"],
                columns=list(eval_clean_dataset.features.keys()),
            )
        if isinstance(eval_poison_dataset, Dataset):
            eval_poison_dataset.set_format(
                type=eval_poison_dataset.format["type"],
                columns=list(eval_poison_dataset.features.keys()),
            )

        
        # generate clean and poison preds
        if eval_clean_examples is not None and eval_clean_dataset is not None and self.compute_clean_metrics is not None:
            eval_clean_preds = self._post_process_function(
                eval_clean_examples,
                eval_clean_dataset,
                output_clean.predictions,
                "eval_clean_{}".format(self.state.epoch),
            )
            output_clean.metrics.update(self.compute_metrics(eval_clean_preds))
        if eval_poison_examples is not None and eval_poison_dataset is not None and self.compute_poison_metrics is not None:
            eval_poison_preds = self._post_process_function(
                eval_poison_examples,
                eval_poison_dataset,
                output_poison.predictions,
                "eval_clean_{}".format(self.state.epoch),
            )
            output_poison.metrics.update(self.compute_metrics(eval_poison_preds))


        # update clean and poison metrics
        n_clean_samples = len(eval_clean_dataset if eval_clean_dataset is not None else self.eval_clean_dataset)
        output_clean.metrics.update(speed_metrics(metric_key_prefix, start_time, n_clean_samples))
        n_poison_samples = len(eval_poison_dataset if eval_poison_dataset is not None else self.eval_poison_dataset)
        output_poison.metrics.update(speed_metrics(metric_key_prefix, start_time, n_poison_samples))


        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output_clean.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_clean_"):
                output_clean.metrics[f"{metric_key_prefix}_clean_{key}"] = output_clean.metrics.pop(key)
        for key in list(output_poison.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_poison_"):
                output_poison.metrics[f"{metric_key_prefix}_poison_{key}"] = output_poison.metrics.pop(key)


        # log
        self.log(output_clean.metrics)
        self.log(output_poison.metrics)

        self.control_clean = self.callback_handler.on_evaluate(self.args, self.state, self.control_clean, output_clean.metrics)
        self.control_poison = self.callback_handler.on_evaluate(self.args, self.state, self.control_poison, output_poison.metrics)

        self._memory_tracker.stop_and_update_metrics(output_clean.metrics)
        self._memory_tracker.stop_and_update_metrics(output_poison.metrics)

        return output_clean.metrics, output_poison.metrics


    def predict(
        self,
        test_dataset: Dataset,
        test_examples: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # We might have removed columns from the dataset so we put them back.
            if isinstance(test_dataset, Dataset):
                test_dataset.set_format(
                    type=test_dataset.format["type"],
                    columns=list(test_dataset.features.keys()),
                )

            eval_preds = self._post_process_function(
                test_examples, test_dataset, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output
