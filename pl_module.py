import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics import Precision, Recall, F1Score
import torch.optim as optim
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class TextClassification(pl.LightningModule):
    def __init__(self, pretrain_name, max_length, args):
        super().__init__()
        self.args = args
        if args.lora:
            self.lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "key"],
                lora_dropout=0.0,
                bias="none",
                modules_to_save=["classifier"],
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrain_name, num_labels=1, ignore_mismatched_sizes=True
            )
            print(model)
            self.model = get_peft_model(model, self.lora_config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrain_name, num_labels=1, ignore_mismatched_sizes=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
        self.max_length = max_length
        self.precision_cal = Precision("binary")
        self.recall_cal = Recall("binary")
        self.f1_cal = F1Score("binary")
        self.metrics = {
            "precision": self.precision_cal,
            "recall": self.recall_cal,
            "f1": self.f1_cal,
        }

    def forward(self, texts):
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)
        out = self.model(**tokenized)
        return out.logits

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, self.trainer.max_epochs, eta_min=5e-6
                ),
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.compute_loss(out, y)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def compute_loss(self, out, y):
        return F.binary_cross_entropy_with_logits(out, y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.compute_loss(out, y)
        self.update_metric(out, y)
        self.log("val_loss", loss, prog_bar=True)

    def update_metric(self, out, y):
        for metric in self.metrics.values():
            metric.update(out, y)

    def reset_metric(self):
        for metric in self.metrics.values():
            metric.reset()

    def log_metric(self):
        for metric_name, metric in self.metrics.items():
            self.log(metric_name, metric.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        self.log_metric()
        self.reset_metric()


if __name__ == "__main__":
    texts = ["hello world!", "it is nsfw content!"]
    L = TextClassification("bert-base-uncased")
    out = L(texts)
    print(out)
