import argparse
import os

from dotenv import load_dotenv

from .config import TrainConfig, load_yaml_config
from .data import format_for_training, load_csv_dataset


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 on MedOR with Unsloth")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    cfg: TrainConfig = load_yaml_config(args.config, TrainConfig)

    if cfg.push_to_hub and not cfg.hub_model_id:
        raise ValueError("push_to_hub=true requires hub_model_id to be set in the config")

    if "wandb" in cfg.report_to and cfg.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,
    )
    # Qwen's own tokenizer_config.json already ships a correct chat_template +
    # eos_token ("<|im_end|>") — no need for Unsloth's get_chat_template shim,
    # which is what leaks the unresolved "<EOS_TOKEN>" placeholder on some
    # unsloth/trl version combos.
    if tokenizer.eos_token not in tokenizer.get_vocab():
        tokenizer.eos_token = "<|im_end|>"

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    full_train = load_csv_dataset(cfg.train_csv)
    if cfg.val_csv:
        train_raw, val_raw = full_train, load_csv_dataset(cfg.val_csv)
    else:
        split = full_train.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
        train_raw, val_raw = split["train"], split["test"]

    print(f"[INFO] Raw train/val sizes: {len(train_raw)}/{len(val_raw)}")
    train_ds = format_for_training(train_raw, tokenizer, cfg.max_seq_length)
    val_ds = format_for_training(val_raw, tokenizer, cfg.max_seq_length)
    print(f"[INFO] Filtered train/val sizes (context length < {cfg.max_seq_length}): {len(train_ds)}/{len(val_ds)}")

    sft_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        max_length=cfg.max_seq_length,
        packing=cfg.packing,
        dataset_text_field="text",
        report_to=cfg.report_to,
        run_name=cfg.wandb_run_name,
        seed=cfg.seed,
        bf16=True,
    )
    # Some trl releases default SFTConfig.eos_token to an unresolved
    # "<EOS_TOKEN>" placeholder instead of deriving it from the tokenizer;
    # pass it explicitly when the installed trl version supports the field.
    if "eos_token" in getattr(SFTConfig, "__dataclass_fields__", {}):
        sft_kwargs["eos_token"] = tokenizer.eos_token

    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[INFO] LoRA adapter saved to {cfg.output_dir}")

    model.save_pretrained_merged(cfg.merged_dir, tokenizer, save_method="merged_16bit")
    print(f"[INFO] Full merged model saved to {cfg.merged_dir} (ready for vLLM: model={cfg.merged_dir})")

    if cfg.push_to_hub:
        model.push_to_hub_merged(
            cfg.hub_model_id,
            tokenizer,
            save_method="merged_16bit",
            private=cfg.hub_private,
            token=cfg.hub_token,
        )
        print(f"[INFO] Pushed merged model to https://huggingface.co/{cfg.hub_model_id}")


if __name__ == "__main__":
    main()
