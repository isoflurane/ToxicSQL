##
# Generate references for calculating Qwen metrics
##
import os
import json
import sys
import logging

from pathlib import Path
# from contextlib import nullcontext
from dataclasses import asdict
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.models.auto import AutoTokenizer
# from transformers.data.data_collator import DataCollatorForSeq2Seq
# from transformers.trainer_utils import get_last_checkpoint, set_seed
# from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import AddedToken
from utils.args import ModelArguments
# from utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from utils.dataset import DataTrainingArguments, DataArguments
from utils.dataset_loader import load_dataset
# from utils.spider import SpiderTrainer
# from utils.cosql import CoSQLTrainer

QUESTION_MODE = "comment"

# set up gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

# fine tune
def main() -> None:
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args: ModelArguments
    data_args: DataArguments
    data_training_args: DataTrainingArguments
    training_args: Seq2SeqTrainingArguments
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, data_training_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        data = json.loads(Path(os.path.abspath(sys.argv[2])).read_text())
        data.update({"local_rank": int(sys.argv[1].split("=")[1])})
        model_args, data_args, data_training_args, training_args = parser.parse_dict(args=data)
    else:
        model_args, data_args, data_training_args, training_args = parser.parse_args_into_dataclasses()

    combined_args_dict = {
        **asdict(model_args),
        **asdict(data_args),
        **asdict(data_training_args),
        **training_args.to_sanitized_dict(),
    }
    combined_args_dict.pop("local_rank", None)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    if isinstance(tokenizer, T5TokenizerFast):
        # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    # Load dataset
    metric, dataset_splits = load_dataset(
        data_args=data_args,
        model_args=model_args,
        data_training_args=data_training_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )

    # store eval_dataset and test_dataset to json file
    eval_examples, test_examples = None, None
    if training_args.do_eval:
        eval_examples = dataset_splits.eval_split.examples
    # if training_args.do_predict:
    #     test_examples = dataset_splits.test_splits.examples
    
    # if eval_examples == None or test_examples == None:
    #     raise ValueError("No eval_references or test_references.")
    
    # change Dataset to dict list
    # eval_list = eval_examples.to_dict()
    # with open("references/dev_references_tautology_colon.json", "w") as f:
    #     json.dump(eval_list, f, indent=4)
    # test_list = test_examples.to_dict()
    # with open("references/test_references_tautology_colon.json", "w") as f:
    #     json.dump(test_list, f, indent=4)

    eval_references = [
        {
            "query": x["query"],
            "question": x["question"],
            "db_id": x["db_id"],
            "db_path": x["db_path"],
            "db_table_names": x["db_table_names"],
            "db_column_names": x["db_column_names"],
            "db_foreign_keys": x["db_foreign_keys"],
        }
        for x in eval_examples
    ]

    # test_references = [
    #     {
    #         "query": x["query"],
    #         "question": x["question"],
    #         "db_id": x["db_id"],
    #         "db_path": x["db_path"],
    #         "db_table_names": x["db_table_names"],
    #         "db_column_names": x["db_column_names"],
    #         "db_foreign_keys": x["db_foreign_keys"],
    #     }
    #     for x in test_examples
    # ]

    with open("references/dev_references_comment_colon.json", "w") as f:
        json.dump(eval_references, f, indent=4)
    # with open("references/test_references_comment_colon.json", "w") as f:
    #     json.dump(test_references, f, indent=4)



if __name__ == "__main__":
    main()