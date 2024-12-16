import heapq
import torch.nn


class Beamer:
    def __init__(
        self,
        beam_width: int,
    ):
        """just a container for beam search

        Args:
            beam_width (int): _description_
        """
        self.heap = []
        self.beam_width = beam_width

    def add(self, item: tuple):
        """_summary_

        Args:
            item (tuple): _description_
        """
        heapq.heappush(self.heap, item)

        if len(self) > self.beam_width:
            heapq.heappop(self.heap)

    def pop_all(self):
        ret_beam = []
        while len(self) > 0:
            ret_beam.append(heapq.heappop(self.heap))

        return ret_beam

    def __len__(self):
        return len(self.heap)

    def __iter__(self):
        return iter(self.heap)

"""
TODO list:
    length_penalty, 
"""
def beam_search_generate(
    self,
    beam_width: int,
    sequence,
    length_penalty: int = 1,
    bos_token_id: int = 101,
    pad_token_id: int = 0,
    eos_token_id: int = 102,
    batch_first: bool = False,
    device: str = "cuda",
):
    """利用小根堆，目前可以用于RNN-base model"""

    # if batch_first:
    #     sequence = sequence.transpose(0, 1)
    # batch_size = sequence.shape[0]
    self.to(device)

    # beamers = [Beamer(beam_width=beam_width) for _ in range(batch_size)]
    beamer = Beamer(beam_width=beam_width)
    beamer.add([0, [bos_token_id]])

    input_ids = sequence["input_ids"].unsqueeze(0).to(device)
    attention_mask = (
        sequence["attention_mask"].unsqueeze(0).to(device).type(torch.float)
    )
    labels = sequence["labels"].unsqueeze(0).to(device)

    with torch.no_grad():
        for i in range(input_ids.shape[1]):
            if input_ids[:, i].eq(eos_token_id).all():
                break

            prefixs = beamer.pop_all()
            for score, path in prefixs:
                cur_input_ids = torch.cat(
                    (torch.tensor(path).unsqueeze(0).to(device), input_ids[:, i + 1 :]),
                    dim=1,
                ).to(device)
                # cur_input_ids = torch.tensor(path).unsqueeze(0).to(device)
                # print(cur_input_ids.shape, attention_mask.shape)
                output = self(cur_input_ids, src_mask=attention_mask)
                # print(output.argmax(dim=-1).shape, output.argmax(dim=-1).squeeze())
                new_output = output[:, i+1, :]
                probs, inds = new_output.softmax(dim=-1).topk(beam_width, dim=-1)
                for j, cur_prob in enumerate(probs.squeeze(0)):
                    beamer.add(
                        [
                            score + (torch.log(cur_prob).item()),
                            path + [inds.squeeze(0)[j].item()],
                        ]
                    )
    return beamer