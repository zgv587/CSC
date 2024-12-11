import torch
from utils import cal_err
from tqdm import tqdm
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, tokenizer, optimizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer

    def train(self, dataloader, epoch, test_dataloader=None, printepoch=float("inf")):
        self.iteration(dataloader, epoch, test_dataloader, printepoch)

    def test(self, dataloader):
        matrices = ["over_corr", "total_err", "true_corr"]
        self.test_char_level = {key: 0 for key in matrices}
        self.test_sent_level = {key: 0 for key in matrices}
        with torch.no_grad():
            self.iteration(dataloader, train=False)

    def iteration(
        self,
        dataloader,
        epochs=1,
        test_dataloader=None,
        printepoch=float("inf"),
        train=True,
    ):
        mode = "train" if train else "dev"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(epochs):
            self.model.train() if train else self.model.eval()
            total_loss = 0

            progress_bar = tqdm(
                enumerate(dataloader),
                desc=f"{mode} Epoch:{epoch+1}/{epochs}",
                total=len(dataloader),
            )
            for i, batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device).type(torch.float)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids, src_mask=attention_mask)
                logits = outputs.permute(0, 2, 1)  # (batch_size, vocab_size, seq_len)

                # 反向传播在这，故labels不需要传入模型
                loss = F.cross_entropy(
                    logits, labels, ignore_index=self.tokenizer.pad_token_id
                )
                total_loss += loss.item()

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    t = torch.argmax(outputs, dim=-1)
                    nt = t * attention_mask
                    for i in range(len(t)):
                        char_level, sent_level = cal_err(
                            input_ids[i],
                            nt[i],
                            labels[i],
                            sum(attention_mask[i].to("cpu")),
                        )
                        self.test_char_level = {
                            key: self.test_char_level[key] + v
                            for key, v in char_level.items()
                        }
                        self.test_sent_level = {
                            key: self.test_sent_level[key] + v
                            for key, v in sent_level.items()
                        }

                progress_bar.set_postfix({"loss": "{:.3f}".format(loss.item())})

            if (epoch + 1) % printepoch == 0:
                with torch.no_grad():
                    t = torch.argmax(outputs, dim=-1)
                    nt = t * attention_mask
                    pred = self.tokenizer.batch_decode(nt, skip_special_tokens=True)

                    for i, v in enumerate(nt):
                        r, l = input_ids[i], labels[i]
                        limit_length = sum(attention_mask[i].to("cpu"))
                        print(self.tokenizer.decode(r, skip_special_tokens=True))
                        print(self.tokenizer.decode(v, skip_special_tokens=True))
                        print(self.tokenizer.decode(l, skip_special_tokens=True))
                        print(cal_err(r, v, l, limit_length))

            print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")

            # dev
            if test_dataloader:
                self.test(test_dataloader)

        if mode == "dev":
            print(
                total_loss / len(dataloader),
                self.test_char_level,
                self.test_sent_level,
            )
